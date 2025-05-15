import torch
from transformers import pipeline
from huggingface_hub import login
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from urllib.parse import quote_plus
import json
import os
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import threading
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# 初始化語言模型管道
pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 初始化嵌入模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Trajectory系統 - 反思機制的核心
class TrajectorySystem:
    def __init__(self, file_path="trajectory_data.json"):
        self.file_path = file_path
        self.data = self.load_data()
    
    def load_data(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except:
                return {"channel_preferences": {"liked": {}, "disliked": {}}}
        else:
            return {"channel_preferences": {"liked": {}, "disliked": {}}}
    
    def save_data(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f)
    
    def update_channel_preference(self, channel, preference):
        """更新頻道偏好 (喜歡或不喜歡)"""
        if preference == "liked":
            if channel in self.data["channel_preferences"]["liked"]:
                self.data["channel_preferences"]["liked"][channel] += 1
            else:
                self.data["channel_preferences"]["liked"][channel] = 1
        elif preference == "disliked":
            if channel in self.data["channel_preferences"]["disliked"]:
                self.data["channel_preferences"]["disliked"][channel] += 1
            else:
                self.data["channel_preferences"]["disliked"][channel] = 1
        self.save_data()
    
    def get_popular_liked_channels(self, limit=5):
        """取得最受歡迎的頻道"""
        channels = self.data["channel_preferences"]["liked"]
        return sorted(channels.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_popular_disliked_channels(self, limit=5):
        """取得最不受歡迎的頻道"""
        channels = self.data["channel_preferences"]["disliked"]
        return sorted(channels.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_statistics(self):
        """取得統計資訊"""
        liked = self.data["channel_preferences"]["liked"]
        disliked = self.data["channel_preferences"]["disliked"]
        return {
            "total_liked_channels": len(liked),
            "total_disliked_channels": len(disliked),
            "top_liked": self.get_popular_liked_channels(3),
            "top_disliked": self.get_popular_disliked_channels(3)
        }

# RAG系統 - 文檔處理與檢索
class RAGSystem:
    def __init__(self, collection_name="study_materials", db_directory="./chroma_db"):
        # 設定文本分割參數
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # 設定ChromaDB向量資料庫
        self.client = chromadb.Client(Settings(
            persist_directory=db_directory,
            anonymized_telemetry=False
        ))
        
        # 創建或連接集合
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
    
    def process_pdf(self, file_path):
        """處理PDF文件並分塊"""
        chunks_with_metadata = []
        try:
            doc = fitz.open(file_path)
            text_by_section = {}
            current_section = "默認章節"
            
            # 提取文檔結構與文本
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # 尋找標題模式
                headers = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
                
                if headers:
                    for header in headers:
                        current_section = header
                        if current_section not in text_by_section:
                            text_by_section[current_section] = []
                
                if current_section not in text_by_section:
                    text_by_section[current_section] = []
                
                text_by_section[current_section].append(text)
            
            # 分割每個章節的文本
            for section, texts in text_by_section.items():
                combined_text = " ".join(texts)
                
                # 基於字符數進行分塊
                chunks = []
                start = 0
                while start < len(combined_text):
                    end = start + self.chunk_size
                    # 如果不是最後一塊，尋找句號作為斷點
                    if end < len(combined_text):
                        # 向後尋找句號
                        while end < len(combined_text) and combined_text[end] not in ['.', '!', '?', '\n']:
                            end += 1
                        if end < len(combined_text):
                            end += 1  # 包含句號
                    
                    chunks.append(combined_text[start:end])
                    start = end - self.chunk_overlap
                
                # 添加元數據
                for i, chunk in enumerate(chunks):
                    chunks_with_metadata.append({
                        "text": chunk,
                        "metadata": {
                            "source": file_path,
                            "title": os.path.basename(file_path),
                            "section": section,
                            "chunk_id": i
                        }
                    })
            
            return chunks_with_metadata
        except Exception as e:
            print(f"處理PDF時出錯: {str(e)}")
            return []
    
    def extract_image_descriptions(self, pdf_path):
        """使用 Perplexity API 分析 PDF 中的圖片，回傳 AI 生成的描述。"""
        descriptions = {}
        api_key = os.getenv("PERPLEXITY_API_KEY")
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images()):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # 將圖片轉為 Base64 Data URI
                    mime = base_image["ext"].lower()
                    data_uri = (
                        f"data:image/{mime};base64,"
                        f"{base64.b64encode(image_bytes).decode()}"
                    )
                    # 建構 Perplexity API 請求
                    payload = {
                        "model": "sonar-pro",
                        "stream": False,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "請描述這張圖片的內容。"},
                                    {"type": "image_url", "image_url": {"url": data_uri}}
                                ]
                            }
                        ]
                    }
                    resp = requests.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    resp.raise_for_status()
                    desc = resp.json()["choices"][0]["message"]["content"]
                    descriptions[f"page_{page_num}_img_{img_index}"] = desc
        except Exception as e:
            print(f"處理圖片時出錯: {e}")
        return descriptions
    
    def add_to_vector_db(self, chunks_with_metadata):
        """將文本塊添加到向量數據庫"""
        ids = []
        texts = []
        metadatas = []
        
        for chunk_data in chunks_with_metadata:
            unique_id = f"{chunk_data['metadata']['title']}_{chunk_data['metadata']['section']}_{chunk_data['metadata']['chunk_id']}"
            ids.append(unique_id)
            texts.append(chunk_data['text'])
            metadatas.append(chunk_data['metadata'])
        
        # 計算嵌入向量
        embeddings = [embedding_model.encode(text).tolist() for text in texts]
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return len(ids)
    
    def query_vector_db(self, query_text, n_results=3):
        """根據查詢檢索相關文本塊"""
        query_embedding = embedding_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def generate_executable_steps(self, query, retrieved_text):
        """根據檢索結果生成可執行步驟"""
        prompt = f"""
        我是一個學習助手，需要根據相關資料生成明確的學習步驟。

        學習主題: {query}
        
        參考檢索結果:
        {retrieved_text}
        
        請生成至少8個具體、可執行的學習步驟，格式如下:
        
        1. [清晰的第一步動作]
        2. [第二步動作]
        ...
        
        每個步驟應該具體、可行，並說明目標和預期結果。
        """
        
        outputs = pipe(prompt, max_new_tokens=1000)
        generated_text = outputs[0]['generated_text']
        
        # 提取生成的步驟
        steps = re.findall(r'\d+\.\s+[^\n]+', generated_text)
        
        return "\n".join(steps) if steps else generated_text

def driver_config():
    """配置Selenium WebDriver"""
    options = Options()
    options.add_argument("disable-blink-features=AutomationControlled")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    
    driver = webdriver.Chrome(options=options)
    return driver

def get_courses(subject, preferred_channels=None, excluded_channels=None):
    """根據主題和頻道偏好抓取YouTube課程"""
    # 修改搜尋查詢以包含頻道偏好
    search_query = f"{subject} programming courses"
    
    # 添加偏好頻道
    preferred_channels_list = []
    if preferred_channels:
        preferred_channels_list = [ch.strip() for ch in preferred_channels.split(',') if ch.strip()]
        for channel in preferred_channels_list:
            trajectory.update_channel_preference(channel, "liked")
            # YouTube不直接支援頻道包含操作，但我們會在結果中優先顯示這些頻道
    
    # 添加排除頻道
    excluded_channels_list = []
    if excluded_channels:
        excluded_channels_list = [ch.strip() for ch in excluded_channels.split(',') if ch.strip()]
        for channel in excluded_channels_list:
            trajectory.update_channel_preference(channel, "disliked")
            search_query += f" -channel:{channel}"
    
    driver = driver_config()
    
    url = f'https://www.youtube.com/results?search_query={quote_plus(search_query)}'
    driver.get(url)
    time.sleep(5)  # 等待頁面載入
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    videos = soup.find_all('a', {'id': 'video-title'})
    courses = []
    
    for video in videos:
        title = video.get('title')
        video_url = 'https://www.youtube.com' + video.get('href')
        
        # 嘗試提取頻道名稱
        try:
            parent_div = video.find_parent('div', {'id': 'dismissible'})
            channel_element = parent_div.find('a', {'class': 'yt-simple-endpoint style-scope yt-formatted-string'})
            channel = channel_element.text if channel_element else "Unknown Channel"
        except:
            channel = "Unknown Channel"
        
        courses.append((title, video_url, channel))
    
    # 根據喜好頻道進行排序
    if preferred_channels_list:
        def is_preferred(course):
            channel = course[2]
            return any(preferred in channel for preferred in preferred_channels_list)
        
        courses = sorted(courses, key=lambda x: (not is_preferred(x), x[0]))
        
    driver.quit()
    return courses

def generate_study_plan(subject, courses, use_rag=True):
    """整合RAG與學習計劃生成"""
    course_titles = "\n".join([f"Title: {title} - Channel: {channel}" for title, url, channel in courses[:5]])
    
    # 使用RAG檢索相關資料
    retrieved_content = ""
    if use_rag:
        results = rag_system.query_vector_db(subject)
        if results and results['documents'] and results['documents'][0]:
            retrieved_content = "\n\n".join(results['documents'][0])
            
    # 結合檢索結果生成學習計劃
    prompt = f"""
    我是一個學習助手，正在為學習 {subject} 制定計劃。
    
    可用課程:
    {course_titles}
    """
    
    if retrieved_content:
        prompt += f"""
    相關知識庫資料:
    {retrieved_content}
    """
        
    prompt += f"""
    請生成一個10週的學習計劃，包含:
    - 每週目標與重點
    - 推薦的學習資源
    - 實踐練習與專案
    - 學習進度評估方式
    """
    
    # 生成學習計劃
    outputs = pipe(prompt, max_new_tokens=2000)
    study_plan = outputs[0]['generated_text']
    
    # 生成可執行步驟
    executable_steps = ""
    if use_rag and retrieved_content:
        executable_steps = rag_system.generate_executable_steps(subject, retrieved_content)
        
    return study_plan, executable_steps

def create_study_plan_with_preferences(subject, preferred_channels=None, excluded_channels=None, use_rag=True):
    """整合爬蟲與語言模型生成計劃的函數"""
    # 從軌跡記錄中獲取熱門頻道偏好
    popular_liked = trajectory.get_popular_liked_channels()
    popular_disliked = trajectory.get_popular_disliked_channels()
    
    # 如果沒有提供偏好，使用熱門偏好
    if not preferred_channels and popular_liked:
        suggested_preferred = ','.join([channel for channel, count in popular_liked])
    else:
        suggested_preferred = preferred_channels
    
    if not excluded_channels and popular_disliked:
        suggested_excluded = ','.join([channel for channel, count in popular_disliked])
    else:
        suggested_excluded = excluded_channels
    
    # 獲取課程
    courses = get_courses(subject, suggested_preferred, suggested_excluded)
    
    # 生成學習計劃，使用RAG增強
    study_plan, executable_steps = generate_study_plan(subject, courses, use_rag)
    
    # 準備輸出
    output = "===== 學習計劃已生成 =====\n\n"
    output += f"主題: {subject}\n"
    
    if suggested_preferred:
        output += f"偏好頻道: {suggested_preferred}\n"
    if suggested_excluded:
        output += f"排除頻道: {suggested_excluded}\n"
    
    output += "\n可用課程:\n"
    for i, (title, url, channel) in enumerate(courses[:10], 1):
        output += f"{i}. 標題: {title}\n   頻道: {channel}\n   URL: {url}\n\n"
    
    output += "\n建議學習計劃:\n"
    output += study_plan
    
    # 加入RAG生成的可執行步驟
    if executable_steps:
        output += "\n\n===== 可執行學習步驟 =====\n"
        output += executable_steps
    
    # 加入統計資訊作為反思
    stats = trajectory.get_statistics()
    output += "\n\n===== 系統統計資訊 =====\n"
    output += f"總共追蹤的偏好頻道數量: {stats['total_liked_channels']}\n"
    output += f"總共排除的頻道數量: {stats['total_disliked_channels']}\n"
    
    if stats['top_liked']:
        output += "\n最受歡迎的頻道:\n"
        for channel, count in stats['top_liked']:
            output += f"- {channel} (被喜歡 {count} 次)\n"
    
    if stats['top_disliked']:
        output += "\n最不受歡迎的頻道:\n"
        for channel, count in stats['top_disliked']:
            output += f"- {channel} (被排除 {count} 次)\n"
    
    return output

# 使用者介面
class StudyPlannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("程式學習計劃產生器")
        self.root.geometry("900x700")
        
        # 創建分頁控制
        self.tab_control = ttk.Notebook(root)
        
        # 建立主要學習計畫頁面
        self.main_frame = ttk.Frame(self.tab_control)
        self.rag_frame = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.main_frame, text="學習計畫生成")
        self.tab_control.add(self.rag_frame, text="RAG 知識管理")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # 設置主要學習計畫頁面
        self.setup_main_page()
        
        # 設置RAG管理頁面
        self.setup_rag_page()
    
    def setup_main_page(self):
        # 建立主框架
        main_frame = tk.Frame(self.main_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 學習主題輸入
        subject_frame = tk.Frame(main_frame)
        subject_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(subject_frame, text="學習主題:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.subject_entry = tk.Entry(subject_frame, width=40, font=("Arial", 12))
        self.subject_entry.pack(side=tk.LEFT, padx=10)
        self.subject_entry.insert(0, "Python")
        
        # RAG開關
        self.use_rag_var = tk.BooleanVar(value=True)
        self.use_rag_check = tk.Checkbutton(subject_frame, text="使用RAG增強", variable=self.use_rag_var, font=("Arial", 11))
        self.use_rag_check.pack(side=tk.LEFT, padx=10)
        
        # 頻道偏好
        channel_frame = tk.LabelFrame(main_frame, text="頻道偏好設定", font=("Arial", 12, "bold"), padx=10, pady=10)
        channel_frame.pack(fill=tk.X, pady=10)
        
        preference_frame = tk.Frame(channel_frame)
        preference_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(preference_frame, text="偏好頻道 (以逗號分隔):", font=("Arial", 11)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.preferred_channels = tk.Entry(preference_frame, width=50, font=("Arial", 11))
        self.preferred_channels.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(preference_frame, text="排除頻道 (以逗號分隔):", font=("Arial", 11)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.excluded_channels = tk.Entry(preference_frame, width=50, font=("Arial", 11))
        self.excluded_channels.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # 獲取推薦按鈕
        recommend_frame = tk.Frame(channel_frame)
        recommend_frame.pack(fill=tk.X, pady=5)
        
        self.recommend_button = tk.Button(recommend_frame, text="取得熱門頻道推薦", command=self.load_recommendations, font=("Arial", 11))
        self.recommend_button.pack(side=tk.RIGHT, padx=5)
        
        # 生成按鈕
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.generate_button = tk.Button(
            button_frame, 
            text="生成學習計劃", 
            command=self.generate_plan,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        )
        self.generate_button.pack(pady=10)
        
        # 進度指示器
        self.progress_frame = tk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Arial", 11))
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="indeterminate")
        
        # 輸出區域
        output_frame = tk.LabelFrame(main_frame, text="生成結果", font=("Arial", 12, "bold"), padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Arial", 11))
        self.output_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_rag_page(self):
        """設置RAG管理頁面"""
        rag_main = tk.Frame(self.rag_frame)
        rag_main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 文件上傳區域
        add_frame = tk.LabelFrame(rag_main, text="添加資料到知識庫", font=("Arial", 12, "bold"))
        add_frame.pack(fill=tk.X, pady=10)
        
        self.add_pdf_button = tk.Button(
            add_frame,
            text="上傳PDF文件",
            command=self.add_pdf_to_db,
            font=("Arial", 11),
            bg="#007BFF",
            fg="white",
            padx=10
        )
        self.add_pdf_button.pack(padx=10, pady=10)
        
        # 知識庫查詢區域
        query_frame = tk.LabelFrame(rag_main, text="知識檢索測試", font=("Arial", 12, "bold"))
        query_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(query_frame, text="輸入查詢:", font=("Arial", 11)).pack(anchor="w", pady=5)
        self.query_entry = tk.Entry(query_frame, width=60, font=("Arial", 11))
        self.query_entry.pack(fill=tk.X, pady=5)
        
        self.query_button = tk.Button(
            query_frame,
            text="檢索知識",
            command=self.query_knowledge,
            font=("Arial", 11)
        )
        self.query_button.pack(anchor="e", pady=5)
        
        self.query_result = scrolledtext.ScrolledText(query_frame, height=15, font=("Arial", 11))
        self.query_result.pack(fill=tk.BOTH, expand=True)
    
    def add_pdf_to_db(self):
        """添加PDF到向量數據庫"""
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return
        
        # 顯示進度
        self.query_result.delete(1.0, tk.END)
        self.query_result.insert(tk.END, f"正在處理 {os.path.basename(file_path)}...\n")
        self.root.update_idletasks()
        
        # 使用線程執行處理以避免界面凍結
        thread = threading.Thread(target=self.process_pdf_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def process_pdf_thread(self, file_path):
        try:
            # 處理PDF
            chunks = rag_system.process_pdf(file_path)
            if chunks:
                # 添加到向量數據庫
                count = rag_system.add_to_vector_db(chunks)
                
                # 更新UI
                self.root.after(0, self.update_rag_ui, f"成功添加 {count} 個文本塊到知識庫!\n文件: {os.path.basename(file_path)}")
            else:
                self.root.after(0, self.update_rag_ui, f"處理文件時出錯: {os.path.basename(file_path)}")
        except Exception as e:
            self.root.after(0, self.update_rag_ui, f"錯誤: {str(e)}")
    
    def update_rag_ui(self, message):
        """更新RAG界面訊息"""
        self.query_result.delete(1.0, tk.END)
        self.query_result.insert(tk.END, message)
    
    def query_knowledge(self):
        """查詢知識庫"""
        query = self.query_entry.get()
        if not query:
            messagebox.showwarning("警告", "請輸入查詢內容")
            return
        
        # 顯示進度
        self.query_result.delete(1.0, tk.END)
        self.query_result.insert(tk.END, f"正在檢索: {query}...\n")
        self.root.update_idletasks()
        
        # 在線程中執行查詢
        thread = threading.Thread(target=self.query_thread, args=(query,))
        thread.daemon = True
        thread.start()
    
    def query_thread(self, query):
        try:
            # 查詢向量數據庫
            results = rag_system.query_vector_db(query)
            
            if not results or not results['documents'] or not results['documents'][0]:
                self.root.after(0, self.update_rag_ui, "未找到相關內容")
                return
            
            # 格式化結果
            output = f"查詢: {query}\n\n檢索結果:\n\n"
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                output += f"--- 結果 {i} ---\n"
                output += f"來源: {metadata['source']}\n"
                output += f"章節: {metadata['section']}\n"
                output += f"內容片段:\n{doc[:500]}...\n\n"
            
            # 生成可執行步驟
            output += "\n可執行步驟:\n"
            steps = rag_system.generate_executable_steps(query, "\n".join(results['documents'][0]))
            output += steps
            
            # 更新UI
            self.root.after(0, self.update_rag_ui, output)
        except Exception as e:
            self.root.after(0, self.update_rag_ui, f"錯誤: {str(e)}")
    
    def load_recommendations(self):
        """載入熱門頻道推薦"""
        # 獲取熱門偏好
        popular_liked = trajectory.get_popular_liked_channels()
        popular_disliked = trajectory.get_popular_disliked_channels()
        
        if popular_liked:
            self.preferred_channels.delete(0, tk.END)
            self.preferred_channels.insert(0, ','.join([channel for channel, count in popular_liked]))
        
        if popular_disliked:
            self.excluded_channels.delete(0, tk.END)
            self.excluded_channels.insert(0, ','.join([channel for channel, count in popular_disliked]))
            
        if not popular_liked and not popular_disliked:
            messagebox.showinfo("推薦", "目前還沒有足夠的使用資料來提供頻道推薦。請先使用系統幾次來建立資料。")
    
    def generate_plan(self):
        """生成學習計劃"""
        # 獲取輸入值
        subject = self.subject_entry.get()
        preferred = self.preferred_channels.get()
        excluded = self.excluded_channels.get()
        use_rag = self.use_rag_var.get()
        
        if not subject:
            messagebox.showwarning("警告", "請輸入學習主題。")
            return
        
        # 清空輸出並顯示進度
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "正在生成學習計劃，請稍候...\n這可能需要幾分鐘時間，取決於網路速度和模型回應速度。")
        
        self.progress_label.config(text="處理中...")
        self.progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.progress.start()
        self.generate_button.config(state=tk.DISABLED)
        
        # 在單獨的線程中執行生成，避免凍結UI
        thread = threading.Thread(target=self.run_generation, args=(subject, preferred, excluded, use_rag))
        thread.daemon = True
        thread.start()
    
    def run_generation(self, subject, preferred, excluded, use_rag):
        """在背景執行生成任務"""
        try:
            # 調用主函數生成學習計劃
            result = create_study_plan_with_preferences(subject, preferred, excluded, use_rag)
            
            # 更新UI顯示結果
            self.root.after(0, self.update_ui, result)
        except Exception as e:
            self.root.after(0, self.update_ui, f"錯誤: {str(e)}\n\n請檢查網路連接或重試。")
    
    def update_ui(self, result):
        """更新UI顯示結果"""
        self.progress.stop()
        self.progress.pack_forget()
        self.progress_label.config(text="完成!")
        self.generate_button.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)

# 初始化系統
trajectory = TrajectorySystem()
rag_system = RAGSystem()

# 主函數
def main():
    root = tk.Tk()
    app = StudyPlannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

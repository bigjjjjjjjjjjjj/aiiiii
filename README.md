# aiiiii

# RAG-Based Operation Manual Assistant

## 專案概述
本專案實作 Retrieval-Augmented Generation (RAG) 系統，將操作手冊（PDF、掃描圖片、網頁）轉換為可檢索的向量知識庫，並結合 Perplexity API 生成精確、可執行的任務步驟與學習計畫。

## 功能特色
- 文檔處理：支持 PDF/掃描圖片的文字提取與 OCR  
- Chunking：自動依章節及固定長度（500 字元、50 重疊）分塊並保留元資料  
- 向量資料庫：利用 SentenceTransformer 建立 ChromaDB 索引  
- 知識檢索：語義檢索返回最相關文本區塊，並自動生成可執行步驟  
- 學習計畫：結合 YouTube 課程與檢索結果，生成 10 週學習計畫  

## 開發環境
- Python 3.8+  
- PyMuPDF (pymupdf)  
- sentence-transformers  
- chromadb  
- requests  
- selenium, beautifulsoup4  
- tkinter  

## 安裝與設定
1. 克隆專案  
git clone https://github.com/bigjjjjjjjjjjjj/aiiiii

2. cd rag-manual-assistant

3. 建立虛擬環境並安裝套件  
python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt

4. 設定環境變數  
export PERPLEXITY_API_KEY="你的 Perplexity API 金鑰"


## 執行方式
1. 啟動主程式：  
python main.py

2. 在 GUI 中：  
- 切換至 “RAG 知識管理” 分頁，點擊「上傳 PDF 文件」將手冊加入知識庫  
- 輸入查詢並點擊「檢索知識」，查看相關段落與可執行步驟  
- 切換至 “學習計畫生成” 分頁，輸入主題並選擇「使用 RAG 增強」，生成 10 週學習計畫  

## 範例
1. 上傳 `manual.pdf`和 `manual2.pdf`
2. 查詢「如何使用迴圈」  
3. 生成 Python 學習計畫並比較 RAG 開/關的差異  



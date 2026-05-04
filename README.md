# 🎥 YouTube Video Q&A Chatbot

An AI-powered application that allows you to **interact with any YouTube video** by asking questions about its content.

The system processes videos with **English subtitles (including auto-generated captions)** and enables **unlimited Q&A** using a Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Features

- ✅ Works with any YouTube video having English subtitles  
- ✅ Supports **auto-generated captions**  
- ✅ Ask **unlimited questions** after processing  
- ✅ Fast semantic search using vector embeddings  
- ✅ Clean and modular pipeline design  

## 🧠 Pipeline Overview
→ Data Ingestion
→ Preprocessing
→ Text Chunking
→ Embedding Generation
→ Vector Store (FAISS)
→ Retriever
→ User Query
→ LLM Processing
→ Output Parsing

## 🛠️ Tech Stack

- **LangChain Ecosystem**
  - langchain-core
  - langchain-community
  - langchain-text-splitters
- **Embeddings:** HuggingFace
- **Vector Store:** FAISS
- **LLM:** Pollinations AI
- **Frontend:** Streamlit
- **Data Source:** youtube-transcript-api

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

``bash
git clone <your-repo-link>
cd youtube_video_chatbot
pip install -r requirements.txt

Setup Pollinations API (IMPORTANT)

This project uses Pollinations AI as the LLM.

👉 Steps:

Go to: https://pollinations.ai
Generate your API key
Open chatbot.py
Replace the API key with your own
api_key = "your_api_key_here"

replace in the chatbot.py file on line '152'

Finally
Run the Application
'streamlit run app.py'

# Bajaj Finserv PDF Processor API

A FastAPI-based PDF processing and question-answering system that extracts text from PDFs, indexes it using FAISS vector search, and provides intelligent answers using OpenAI's GPT model.

## 🚀 Features

- **PDF Text Extraction**: Supports both PyMuPDF and OCR (EasyOCR) for text extraction
- **Vector Search**: FAISS + BM25 ensemble retrieval for accurate document search
- **AI-Powered Q&A**: OpenAI GPT-3.5-turbo for intelligent answer generation
- **Multi-language Support**: Handles English and Hindi text
- **Fast Processing**: Optimized for sub-30 second response times
- **RESTful API**: Clean FastAPI endpoints with automatic documentation

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: PostgreSQL (Neon DB)
- **Vector Search**: FAISS + BM25 (LangChain)
- **AI Model**: OpenAI GPT-3.5-turbo
- **PDF Processing**: PyMuPDF, EasyOCR, pdf2image
- **Text Processing**: LangChain, Sentence Transformers

## 📋 Prerequisites

- Python 3.12+
- PostgreSQL database (or Neon DB)
- OpenAI API key
- Chrome/Chromium (for browser error detection)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/DivamSanghvi/bajaj_finserv.git
cd bajaj_finserv
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# FastAPI settings
APP_HOST=127.0.0.1
APP_PORT=8000

# Database settings (Neon DB)
POSTGRES_HOST=your-neon-host
POSTGRES_PORT=5432
POSTGRES_DB=your-db-name
POSTGRES_USER=your-db-user
POSTGRES_PASSWORD=your-db-password

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key
```

### 5. Run the Server
```bash
uvicorn main:app --reload
```

The API will be available at: http://127.0.0.1:8000

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Main Endpoints

#### 1. PDF Upload & Processing
```http
POST /pdf/upload_and_process
Content-Type: multipart/form-data

Parameters:
- file: PDF file
- resource_id: string
- project_id: string
```

#### 2. Document Search
```http
POST /pdf/search
Content-Type: application/x-www-form-urlencoded

Parameters:
- query: string
- project_id: string
- resource_ids: string (comma-separated, optional)
```

#### 3. HackRx Q&A System
```http
POST /hackrx/run
Content-Type: application/json

{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months..."
    ]
}
```

## 🔧 Configuration

### PDF Processing Settings
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 100 characters
- **Retrieval**: Top 4 chunks per query
- **Model**: OpenAI GPT-3.5-turbo

### Vector Search Configuration
- **FAISS Weight**: 70% (semantic search)
- **BM25 Weight**: 30% (keyword search)
- **Embeddings**: all-MiniLM-L6-v2

## 📁 Project Structure

```
bajaj_finserv/
├── main.py                 # FastAPI application
├── pdf_processor.py        # PDF processing logic
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── venv/                 # Virtual environment (ignored)
├── temp_pdfs/           # Temporary PDF storage (ignored)
├── uploaded_pdfs/       # Uploaded PDF storage (ignored)
└── vector_stores/       # FAISS index storage (ignored)
```

## 🔍 How It Works

1. **PDF Upload**: PDF is downloaded from URL or uploaded directly
2. **Text Extraction**: PyMuPDF extracts text, falls back to OCR if needed
3. **Chunking**: Text is split into 800-character chunks with overlap
4. **Vectorization**: Chunks are embedded using Sentence Transformers
5. **Indexing**: FAISS + BM25 ensemble for hybrid search
6. **Query Processing**: Questions are processed against indexed content
7. **Answer Generation**: OpenAI GPT generates concise, accurate answers

## 🚀 Performance

- **Response Time**: < 30 seconds for 10 questions
- **Accuracy**: High precision with ensemble retrieval
- **Scalability**: Supports multiple projects and resources
- **Reliability**: Fallback mechanisms for text extraction

## 🔒 Security

- Environment variables for sensitive data
- Virtual environment isolation
- Input validation and sanitization
- Error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the logs for debugging information

---

**Built with ❤️ for Bajaj Finserv** 
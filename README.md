# Bajaj Finserv PDF Processor API

A FastAPI-based PDF processing and question-answering system that extracts text from PDFs, indexes it using FAISS vector search, and provides intelligent answers using OpenAI's GPT model with structured policy analysis.

## 🚀 Features

- **PDF Text Extraction**: Supports both PyMuPDF and OCR (EasyOCR) for text extraction
- **Vector Search**: FAISS + BM25 ensemble retrieval for accurate document search
- **AI-Powered Q&A**: OpenAI GPT-3.5-turbo for intelligent answer generation
- **Structured Policy Analysis**: Multi-stage processing with clause categorization and decision reasoning
- **Multi-language Support**: Handles English and Hindi text
- **Fast Processing**: Optimized for sub-30 second response times
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **Enhanced Analysis**: Detailed policy insights with confidence scoring and risk assessment

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: PostgreSQL (Neon DB)
- **Vector Search**: FAISS + BM25 (LangChain)
- **AI Model**: OpenAI GPT-3.5-turbo
- **PDF Processing**: PyMuPDF, EasyOCR, pdf2image
- **Text Processing**: LangChain, Sentence Transformers
- **Analysis Engine**: Structured JSON parsing with fallback mechanisms

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

#### 3. HackRx Q&A System (Basic)
```http
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer 343c934c163f8f87a6a809c5c79729281f6fdbf03592227539766d3097f11fcd

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
        "Yes, a grace period of thirty days is provided for premium payment...",
        "No, there is a waiting period of thirty-six (36) months..."
    ]
}
```

#### 4. HackRx Detailed Analysis (Enhanced)
```http
POST /hackrx/analyze
Content-Type: application/json
Authorization: Bearer 343c934c163f8f87a6a809c5c79729281f6fdbf03592227539766d3097f11fcd

{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Enhanced Response:**
```json
{
    "analyses": [
        {
            "question": "What is the grace period for premium payment?",
            "answer": "Yes, a grace period of thirty days is provided for premium payment.",
            "decision": {
                "decision": "approved",
                "confidence_score": 0.85,
                "reasoning": "Coverage clause applies. Waiting period of 30 days applies.",
                "risk_factors": [],
                "recommendations": []
            },
            "clauses_analyzed": 3,
            "relevant_clauses": [
                {
                    "clause_id": "Grace Period Clause",
                    "relevance_score": 0.95,
                    "clause_type": "inclusion",
                    "extracted_rules": {
                        "waiting_period_months": 1,
                        "coverage_amount": null,
                        "exclusions_mentioned": [],
                        "conditions_mentioned": ["timely payment"]
                    },
                    "reasoning": "Directly addresses grace period for premium payments"
                }
            ]
        }
    ]
}
```

## 🔧 Configuration

### PDF Processing Settings
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 100 characters
- **Retrieval**: Top 6 chunks per query
- **Model**: OpenAI GPT-3.5-turbo

### Vector Search Configuration
- **FAISS Weight**: 80% (semantic search)
- **BM25 Weight**: 20% (keyword search)
- **Embeddings**: all-MiniLM-L6-v2

### Analysis Engine Settings
- **Clause Types**: inclusion, exclusion, condition, general
- **Decision Rules**: Exclusion Priority → Waiting Period → Approval Logic
- **Confidence Scoring**: 0.0 to 1.0 scale
- **JSON Parsing**: Robust with markdown fallback

## 📁 Project Structure

```
bajaj_finserv/
├── main.py                 # FastAPI application with enhanced analysis
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

### Enhanced Processing Pipeline

1. **PDF Upload**: PDF is downloaded from URL or uploaded directly
2. **Text Extraction**: PyMuPDF extracts text, falls back to OCR if needed
3. **Chunking**: Text is split into 800-character chunks with overlap
4. **Vectorization**: Chunks are embedded using Sentence Transformers
5. **Indexing**: FAISS + BM25 ensemble for hybrid search
6. **Query Processing**: Questions are processed against indexed content
7. **Structured Analysis**: 
   - **Clause Analysis**: Categorize policy sections (inclusion/exclusion/condition)
   - **Rule Extraction**: Extract waiting periods, coverage amounts, conditions
   - **Decision Reasoning**: Apply business rules systematically
   - **Answer Generation**: Generate concise, structured answers
8. **Response Formatting**: Clean JSON with confidence scoring and recommendations

### Decision Making Logic

The system follows a structured decision-making process inspired by professional insurance analysis:

1. **Exclusion Priority**: If any relevant clause is an 'exclusion', decision is 'rejected'
2. **Waiting Period Validation**: Check if waiting periods are met
3. **Pre-existing Conditions**: Evaluate pre-existing condition clauses
4. **Approval Logic**: If no exclusions apply and inclusions exist, decision is 'approved'
5. **Review Required**: If information is insufficient, status is 'requires_review'

## 🚀 Performance

- **Response Time**: < 30 seconds for 10 questions
- **Accuracy**: High precision with ensemble retrieval and structured analysis
- **Scalability**: Supports multiple projects and resources
- **Reliability**: Fallback mechanisms for text extraction and JSON parsing
- **Analysis Depth**: Comprehensive policy insights with confidence scoring

## 🔒 Security

- Environment variables for sensitive data
- Virtual environment isolation
- Input validation and sanitization
- Error handling and logging
- Bearer token authentication for sensitive endpoints

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
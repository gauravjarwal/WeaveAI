# WeaveAI - AI-Powered Knowledge Base Search & Enrichment

WeaveAI is a sophisticated RAG (Retrieval-Augmented Generation) system that allows users to upload documents, search them using natural language, and receive AI-generated answers with completeness detection and automatic enrichment capabilities.

## 📋 Prerequisites

- Python 3.8 or higher

## 🛠️ Installation & Setup

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd WeaveAI

# 2. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python3 main.py
```

### Access Points
- 🌐 **Web Interface**: http://localhost:8000
- 📚 **API Documentation**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

## 🧪 Testing the System (See Usage Guide for more detailed instructions)

1. Open `http://localhost:8000` in your browser
2. Upload sample documents (PDF, DOCX, TXT, or MD)
3. Ask questions about the document content
4. Verify completeness detection and enrichment suggestions
5. Test the auto-enrichment feature
6. Test document management (view and delete files)
7. Test the feedback system

## 📖 Usage Guide

### 1. Upload Documents
- Click the upload area or drag and drop files
- Supported formats: PDF, DOCX, TXT, MD
- Maximum file size: 50MB per file
- Documents are automatically processed and indexed
- View uploaded documents in the expandable "Documents" section

### 2. Ask Questions
- Type your question in natural language
- Click "Search" or press Enter
- Receive AI-generated answers with confidence scores
- Get concise missing information and enrichment suggestions

### 3. Review Results
- **Answer**: AI-generated response based on your documents
- **Confidence**: Reliability score (0-100%)
- **Missing Information**: Brief, generic types of missing information (max 2 items)
- **Enrichment Suggestions**: Concise recommendations for improvement (max 2 items)
- **Sources**: Referenced documents with relevance scores

### 4. Document Management
- Expand the "Documents" section to see all uploaded files
- Delete documents using the red trash icon
- Deleted documents are removed from both the vector database and file system

### 5. Auto-Enrichment
- Click "Auto-Enrich Knowledge Base" after getting search results
- System generates relevant content using OpenAI API
- Creates clean, descriptive filenames (e.g., `auto_enriched_spaceships.txt`)
- Automatically adds enriched content to the knowledge base

### 6. Provide Feedback
- Rate answers using the 5-star system
- Add optional text feedback
- Help improve future responses

## 🚀 Features

### Core Features
- **Document Upload & Management**: Support for PDF, DOCX, TXT, and MD files with file management
- **Natural Language Search**: Ask questions in plain language with semantic search
- **AI-Generated Answers**: Comprehensive responses using retrieved documents
- **Completeness Detection**: AI identifies when information is missing or uncertain
- **Enrichment Suggestions**: Concise recommendations to improve the knowledge base
- **Confidence Scoring**: Reliability assessment for each answer (0-100%)
- **Source Attribution**: Clear references to source documents and chunks

### Advanced Features
- **Auto-Enrichment**: Automatically generates missing information using OpenAI API
- **Document Management**: View all uploaded documents and delete them as needed
- **Answer Quality Rating**: User feedback system for continuous improvement
- **Modern Web Interface**: Responsive, intuitive UI built with Tailwind CSS and Alpine.js
- **Vector Search**: Semantic similarity search using sentence transformers
- **Structured Output**: JSON responses with answer, confidence, missing_info, and suggestions
- **Comprehensive Logging**: Detailed logs for queries, responses, uploads, and enrichment

## 🎯 Design Decisions

### 1. Vector Database Choice
**Decision**: ChromaDB with persistent storage
**Rationale**: 
- Lightweight and easy to deploy
- Excellent Python integration
- Persistent storage without complex setup
- Good performance for moderate-scale applications

### 2. Embedding Model
**Decision**: all-MiniLM-L6-v2
**Rationale**:
- Balanced performance vs. speed
- Reasonable memory footprint (384 dimensions)
- Proven effectiveness for semantic search

### 3. Chunking Strategy
**Decision**: RecursiveCharacterTextSplitter
**Rationale**:
- Preserves semantic coherence
- Optimal for retrieval performance
- Handles various document structures
- Balances context vs. precision

### 4. LLM Integration
**Decision**: OpenAI GPT-4 with structured prompts and strict hallucination prevention
**Rationale**:
- Superior reasoning capabilities
- Excellent instruction following
- JSON output support for structured responses
- Reliable completeness analysis
- Strict controls to prevent AI hallucination

### 5. Auto-Enrichment Strategy
**Decision**: Direct OpenAI API content generation (can be set to different external APIs)
**Rationale**:
- Simplified development and accelerates prototype building
- Consistent content quality
- Better integration with existing OpenAI infrastructure
- Faster response times

### 6. Web Framework
**Decision**: FastAPI + Alpine.js + Tailwind CSS
**Rationale**:
- Fast development and deployment
- Async support for better performance
- Lightweight frontend with reactive features
- Modern, responsive design

## ⚖️ Trade-offs & Limitations

### Simplified Components
1. **Authentication**: No user management system
2. **Database**: File-based storage instead of production database
3. **Caching**: No Redis or advanced caching layer
4. **Monitoring**: Basic logging instead of comprehensive observability
5. **Testing**: Limited unit tests

### Production Considerations
1. **Scalability**: Single-instance deployment
2. **Security**: Basic input validation
3. **Error Handling**: Simplified error responses
4. **Performance**: No query optimization or connection pooling
5. **Deployment**: Local development setup instead of containerization

### AI Limitations
1. **Language Support**: Primarily English-focused
2. **Real-time Features**: Polling instead of WebSockets

### Additional Limitations
1. **Security**: Basic authentication and input validation only
2. **User Feedback Storage**: File-based feedback storage instead of MySQL database
3. **Auto-Enrichment Sources**: Uses OpenAI-generated content instead of real external trusted sources (Wikipedia, arXiv, etc.)
4. **Response Delivery**: Text-only responses without audio streaming capabilities

### Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4 (Azure deployment)
- **Frontend**: HTML5, Tailwind CSS, Alpine.js
- **Document Processing**: PyPDF2, python-docx, langchain text splitters
- **HTTP Client**: httpx with timeout handling
- **Environment Management**: python-dotenv

## 🔧 API Endpoints

### Core Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload documents
- `POST /search` - Search knowledge base
- `POST /feedback` - Submit answer rating
- `GET /stats` - Knowledge base statistics
- `GET /health` - Health check

### Document Management

- `GET /documents` - List all uploaded documents
- `DELETE /documents/{document_id}` - Delete a specific document

### Auto-Enrichment

- `POST /auto-enrich` - Auto-enrich knowledge base with OpenAI-generated content
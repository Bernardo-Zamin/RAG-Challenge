# RAG-Challenge

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot system that enhances Large Language Model responses with relevant context from a knowledge base. The system retrieves information from documents, embeds them, and uses semantic search to supplement the LLM's generation capabilities.

*[Here](https://drive.google.com/file/d/13xGKKqc2F--Zem-1_6S6w_DUNjc6VMFm/view?usp=sharing) is a link to see the video demo!* 


![RAG System Architecture Diagram - Add visualization of your RAG system here](readme/ui-image.png)

## Features
- Document ingestion and preprocessing pipeline
- Vector embedding storage and retrieval
- Semantic similarity search
- Context-aware prompt engineering

## Requirements
- Python 3.11.13
- Docker (for containerized deployment)
- Miniconda (for virtual enviroment)


## Dependencies
Main packages:
- `fastapi==0.116.1` - API server framework
- `uvicorn==0.35.0` - ASGI server for FastAPI
- `python-multipart==0.0.20` - Multipart form data parsing
- `pymupdf==1.26.4` - PDF document processing
- `faiss-cpu==1.12.0` - Vector storage and similarity search
- `sentence-transformers==5.1.0` - Text embedding models
- `requests==2.32.5` - HTTP client for API calls
- `streamlit==1.49.0` - Web interface
- `qdrant-client==1.15.1` - Vector database client
- `tiktoken==0.11.0` - Tokenizer for text processing
- `nltk==3.9.1` - Natural Language Toolkit
- `python-dotenv==1.1.1` - Environment variable management



## Docker Deployment

1. Clone the repository
    ```bash
    git clone https://github.com/Bernardo-Zamin/RAG-Challenge
    cd RAG-Challenge
    ```

2. Start the Docker containers
    ```bash
    docker compose down --remove-orphans -v
    docker compose up --build
    ```

4. Access the services
    - Web UI: http://localhost:8501
    - Vector DataBase Dashboard: http://localhost:6333/dashboard
    - API: http://localhost:8000/docs

 

The docker-compose.yml file sets up multiple containers:
- API server: FastAPI backend for document processing and queries
- Web UI: Streamlit interface for interacting with the RAG system
- Vector Database: Qdrant for storing and retrieving document embeddings
- Ollama: Local LLM server

#### *I recommend using Docker Desktop for easier visualization, launching, and stopping of containers during development*
![Docker Desktop running the RAG application - Add screenshot here](readme/dockerdesktop-image.png)


### API Endpoints
The application exposes the following endpoints:

1. **Start Chat Endpoint**
   - `POST /start_chat`
   - Create a new chat session
   - Returns a unique session_id for document uploads and questions

2. **Document Upload Endpoint**
   - `POST /documents`
   - Upload PDF documents to be processed and indexed for a specific session
   - Requires session_id and PDF files
   - Returns upload statistics (documents indexed, total chunks, indexed points)

3. **Question Endpoint**
   - `POST /question`
   - Send questions to the RAG system and receive augmented responses
   - Requires question text and session_id
   - Returns AI-generated answer with relevant document references


#### Endpoint Details

| Endpoint | Method | Content-Type | Request Body/Parameters |
|----------|--------|--------------|-------------------------|
| `/start_chat` | POST | - | None |
| `/documents` | POST | multipart/form-data | `session_id` (form field), `files` (PDF files) |
| `/question` | POST | application/json | `{"question": "string", "session_id": "string"}` |

## Project Structure
```
RAG-Challenge/
├── Dockerfile             # API container definition
├── Dockerfile.ollama      # Ollama container definition
├── RAG-Challenge/
│   ├── data/
│   │   └── uploaded_pdfs/ # Uploaded PDF storage
│   │       └── BernardoZamin_CV.pdf
│   ├── src/
│   │   ├── api_routes/
│   │   │   └── api_routes.py
│   │   ├── main.py        # FastAPI application entry
│   │   ├── models/
│   │   │   └── models.py  # Pydantic models
│   │   ├── services/      # Core business logic
│   │   │   ├── embeddings.py
│   │   │   ├── ollama_client.py
│   │   │   ├── pdf_parser.py
│   │   │   └── rag_pipeline.py
│   │   └── vector_database/
│   │       └── qdrant_store.py
│   └── streamlit_app/     # Web UI components
│       ├── ui.py
│       └── utils.py
├── README.md              # This file
├── case_files/            # Sample input PDF documents
│   ├── LB5001.pdf
│   ├── MN414_0224.pdf
│   ├── WEG-CESTARI-manual-iom-guia-consulta-rapida-50111652-pt-en-es-web.pdf
│   └── WEG-motores-eletricos-guia-de-especificacao-50032749-brochure-portuguese-web.pdf
├── docker-compose.yml     # Container orchestration
├── readme/                # Documentation assets
│   ├── dockerdesktop-image.png
│   └── ui-image.png
└── requirements.txt       # Python dependencies
```

## Additional Resources
- For additional models to test with this RAG chatbot, visit [Ollama Model Library](https://ollama.com/search)

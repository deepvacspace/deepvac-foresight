# APPUP RAG SYSTEM

## Overview
The APPUP RAG System is a Flask and FastAPI-based application designed to provide a Retrieval-Augmented Generation (RAG) pipeline for chatbot interactions. It integrates with Google Cloud services, OpenAI, and ChromaDB to deliver intelligent responses based on user queries and stored knowledge.

## Features
- **Flask Backend**: Handles user authentication, chatbot configuration, and conversation history.
- **FastAPI Backend**: Provides endpoints for RAG pipeline and web crawling.
- **ChromaDB Integration**: Manages vectorized document storage for similarity search.
- **Google Cloud Integration**: Uses Firestore for chat history and Cloud Storage for ChromaDB backups.
- **OpenAI Integration**: Leverages GPT-based models for generating responses.
- **Dockerized Deployment**: Supports containerized deployment with Docker and Docker Compose.

## Project Structure
```
.
├── appup/                 # Flask application
│   ├── chatbot/           # Chatbot-related views and forms
│   ├── templates/         # HTML templates for the Flask app
│   ├── models.py          # Database models
│   ├── __init__.py        # Flask app initialization
├── fastapi_app/           # FastAPI application
│   ├── main.py            # FastAPI entry point
│   ├── storage.py         # RAG pipeline and ChromaDB integration
│   ├── chat_history.py    # Firestore chat history management
├── migrations/            # Database migrations
├── readable_pages/        # Crawled web pages for processing
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Poetry configuration
├── README.md              # Project documentation
```

## Prerequisites
- Python 3.10+
- Docker and Docker Compose
- Google Cloud credentials (`firebase_key.json`)
- OpenAI API key

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/appup-rag-system.git
cd appup-rag-system
```

### 2. Set Up Environment Variables
Create `.env.dev` and `.env.prod` files with the following variables:
```
CHROMA_BUCKET_NAME=your-bucket-name
CHROMA_BASE_PATH=/path/to/chroma
CHATBOT_SYSTEM_PROMPT=your-prompt
DEEPSEEK_API_KEY=your-openai-api-key
PGHOST=localhost
POSTGRES_DB=appup
PGUSER=postgres
POSTGRES_PASSWORD=your-password
FLASK_ENV=development
FLASK_DEBUG=True
```

### 3. Install Dependencies
Using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
#### With Docker Compose:
```bash
docker-compose up --build
```

#### Without Docker:
Start the Flask app:
```bash
flask run --host=0.0.0.0
```

Start the FastAPI app:
```bash
uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000
```

## Usage

### Flask Endpoints
- `/chatbot/initial-message`: Initializes a chatbot session.
- `/chatbot/message`: Sends a user message to the chatbot.
- `/chatbot/<slug>/configuracion`: Configures chatbot settings.
- `/chatbot/<slug>/historial`: Retrieves conversation history.

### FastAPI Endpoints
- `/query`: Queries the RAG pipeline.
- `/crawl`: Crawls a webpage and extracts content.

## Deployment

### Production with Docker Compose
```bash
docker-compose --profile prod up --build
```

### Nginx Configuration
Ensure the `nginx/conf.d` directory contains the appropriate configuration for reverse proxying Flask and FastAPI services.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, contact Miguel Quiñones at [miguel.aqr99@gmail.com](mailto:miguel.aqr99@gmail.com).
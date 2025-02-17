# EPub RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system that combines vector search capabilities with Large Language Models to provide accurate, context-aware responses to queries about your EPub documents.

## 🌟 Features

- Load and process multiple EPub documents
- Semantic search using Milvus vector database
- Intelligent context retrieval with fallback to LLM knowledge
- Streamlit-based chat interface
- Real-time response streaming
- Conversation history management
- Automatic source attribution (Knowledge Base vs LLM)

## 🔧 Prerequisites

- Python 3.8+
- Milvus server running (version 2.0+) on docker - https://milvus.io/docs/install_standalone-docker.md
- Ollama server with llama3.2:3b model (or another LLM and embedding model)
- Docker (for running Milvus)

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/rchow93/LangchainRAGwithMilvus.git
cd LangchainRAGwithMilvus
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your environment:
   - Update the Milvus URI in both `app.py` and `dataloader.py`
   - Update the Ollama base URL if needed
   - Place your EPub files in the `docs/epub` directory (or modify the path in the dataloader)

4. Load your documents:
```bash
python dataloader.py
```

5. Start the application:
```bash
streamlit run app.py
```

## 📚 System Components

### Data Loader (`dataloader.py`)

The data loader handles:
- Loading EPub documents from a specified directory
- Splitting documents into manageable chunks
- Computing embeddings using Ollama
- Storing documents and embeddings in Milvus

Configuration options:
```python
MILVUS_URI = "http://your.milvus.server:19530"
COLLECTION_NAME = "YourCollection"
```

### Chat Application (`app.py`)

The main application provides:
- Interactive chat interface
- Semantic search in your document collection
- Automatic fallback to LLM knowledge when needed
- Conversation history tracking
- Real-time response streaming

## 🔍 How It Works

1. **Document Processing**:
   - EPub files are loaded and split into chunks
   - Each chunk is embedded using the nomic-embed-text model
   - Embeddings and text are stored in Milvus

2. **Query Processing**:
   - User questions are embedded using the same model
   - Semantic search finds relevant documents
   - Relevance is determined using cosine similarity
   - If relevant context is found, it's used to inform the response
   - If no relevant context exists, the system falls back to LLM knowledge

3. **Response Generation**:
   - Responses clearly indicate their source (Knowledge Base or LLM)
   - All responses are streamed in real-time
   - Conversation history is maintained for context

## ⚙️ Configuration

### Milvus Settings
```python
MILVUS_URI = "http://your.milvus.server:19530"
COLLECTION_NAME = "YourCollection"
```

### Embedding Model
```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://your.ollama.server:11434"
)
```

### Document Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    keep_separator=True
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- Ensure your Milvus server has enough resources to handle your document collection
- Monitor your Ollama server's memory usage with large document collections
- Adjust chunk sizes and overlap based on your specific use case
- The system works best with well-structured EPub documents
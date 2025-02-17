from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from pathlib import Path
from tqdm import tqdm
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Milvus Connection
MILVUS_URI = "http://192.168.xx.xxx:19530"  ####Please set this to your Milvus URI
COLLECTION_NAME = "EpubDocs"  ####Please set this to your collection name

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",   ####Please set this to your Ollama embedding model
    base_url="http://192.168.xx.xxx:11434",    ####Please set this to your Ollama base URL
)

def load_epub_documents(directory: str):
    docs = []
    epub_files = list(Path(directory).rglob("*.epub"))

    for file_path in tqdm(epub_files, desc="Loading EPUB files"):
        try:
            loader = UnstructuredEPubLoader(str(file_path))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading EPUB {file_path}: {e}")
            continue
    return docs

def create_collection():
    connections.connect("default", uri=MILVUS_URI)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Collection for ArgoCD documents")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    collection.load()

def store_in_milvus(documents) -> None:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        keep_separator=True
    )
    docs = text_splitter.split_documents(documents)

    try:
        # Delete existing collection if it exists
        try:
            connections.connect("default", uri=MILVUS_URI)
            Collection(name=COLLECTION_NAME).drop()
        except:
            pass

        create_collection()
        vector_db = Milvus.from_documents(
            documents=docs,
            embedding=embeddings,
            connection_args={"uri": MILVUS_URI, "timeout": 60},
            collection_name=COLLECTION_NAME,
        )
        print(f"Successfully stored {len(docs)} document chunks in Milvus.")
    except Exception as e:
        print(f"Error storing documents in Milvus: {str(e)}")

if __name__ == "__main__":
    try:
        epub_documents = load_epub_documents("./docs/epub")
        if epub_documents:
            print(f"Loaded {len(epub_documents)} documents successfully")
            try:
                store_in_milvus(epub_documents)
            except Exception as e:
                print(f"Failed to store documents in Milvus: {e}")
        else:
            print("No documents were loaded successfully.")
    except Exception as e:
        print(f"Fatal error: {e}")
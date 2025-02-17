import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List


class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.stream_display = self.container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # Update the display with the complete text so far
        self.stream_display.markdown(self.text)

    def get_response(self) -> str:
        return self.text


# Milvus Connection
MILVUS_URI = "http://192.168.xx.xxx:19530"   ####Please set this to your Milvus URI
COLLECTION_NAME = "EpubDocs"      ####Please set this to your collection name

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",   ####Please set this to your Ollama embedding model
    base_url="http://192.168.xx.xxx:11434"    ####Please set this to your Ollama base URL
)

# Initialize Milvus Vector Store
vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name=COLLECTION_NAME
)

# Create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})


def format_chat_history(messages: List[dict]) -> str:
    return "\n".join([
        f"Human: {msg['content']}" if msg['role'] == "user"
        else f"Assistant: {msg['content']}"
        for msg in messages[-8:]
    ])

def augment_query(question: str, chat_history: str) -> str:
    """Augment the question using chat history for better retrieval."""
    if chat_history:
        return f"Based on previous conversation: {chat_history}\nQuestion: {question}"
    return question


def get_relevant_context(question, chat_history):
    """Get relevant context using semantic search and confidence scoring."""
    results = retriever.invoke(question)

    if not results:
        return None, False

    # Analyze semantic similarity and coherence of retrieved documents
    relevant_docs = []
    for doc in results:
        # Get embedding for the question and document
        question_embedding = embeddings.embed_query(question)
        doc_embedding = embeddings.embed_query(doc.page_content)

        # Calculate cosine similarity
        similarity = compute_cosine_similarity(question_embedding, doc_embedding)

        if similarity > 0.7:  # Threshold can be adjusted
            relevant_docs.append((doc, similarity))

    # Sort by similarity score
    relevant_docs.sort(key=lambda x: x[1], reverse=True)

    if relevant_docs:
        # Return only the documents, discard scores
        return [doc for doc, _ in relevant_docs], True

    return None, False


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

template = """
You are an established and experienced technical expert on the subject being asked. Use the following conversation history and context to provide accurate, helpful answers.
Provide very detail and in depth answers to the question being asked. If you don't know the answer based on the given context, just say that you don't know.

Previous Conversation:
{chat_history}

Current Context:
{context}

Current Question: {question}

Answer the question based on the context and previous conversation:"""

prompt = ChatPromptTemplate.from_template(template)


def generate_response(question: str, context: str, chat_history: str, stream_container):
    stream_handler = StreamlitCallbackHandler(stream_container)

    llm = OllamaLLM(
        model="llama3.2:3b",   ####Please set this to your Ollama LLM model
        base_url="http://192.168.xx.xxx:11434",  ####Please set this to your Ollama base URL
        callbacks=[stream_handler],
        stop=["<|eot_id|>"],
    )

    # Get context and check if it's relevant
    context_docs, has_relevant_context = get_relevant_context(question, chat_history)

    if has_relevant_context:
        messages = [
            SystemMessage(content="""You are a helpful AI assistant. When provided with context, 
            use it to answer questions accurately. Start your response with 'Based on the information 
            in our knowledge base: '"""),
            HumanMessage(content=f"""Question: {question}

Context: {[doc.page_content for doc in context_docs]}

Please answer based on the above context.""")
        ]
    else:
        messages = [
            SystemMessage(content="""You are a helpful AI assistant. Answer questions accurately 
            using your general knowledge. Begin your response with 'I could not find the answer 
            to your question in the Milvus Vector Database - but using my LLM knowledge: '"""),
            HumanMessage(content=question)
        ]

    # Get the response
    llm.invoke(messages)
    # Return the complete accumulated text
    return stream_handler.get_response()


def main():
    st.title("EPub RAG Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask me anything!"):
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        chat_history = format_chat_history(st.session_state.messages[:-1])

        with st.chat_message("assistant"):
            # Create a persistent container for the response
            response_container = st.empty()
            with st.spinner("Processing your question..."):
                response = generate_response(question, None, chat_history, response_container)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
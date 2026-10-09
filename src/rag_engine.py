import os
import warnings
from dotenv import load_dotenv

# Suppress non-critical deprecation warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def build_or_load_vectorstore(docs_dir: str = "data/docs", persist_dir: str = "data/processed/chroma_db"):
    """
    Ingest PDFs from data/docs and build or load a local Chroma Vector DB.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing from environment.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[RAG Warning] No PDF documents found in '{docs_dir}'. Place policy PDFs there to test RAG.")
        return None

    print(f"[RAG] Ingesting {len(pdf_files)} document(s) from {docs_dir}...")
    loader = DirectoryLoader(docs_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"[RAG] Successfully stored {len(chunks)} text chunks in ChromaDB ({persist_dir}).")
    return vectorstore


def query_rag_system(user_query: str, vectorstore) -> str:
    """
    Execute semantic search and response generation using GPT-4o-mini via LCEL pipeline.
    """
    if vectorstore is None:
        return "No documents available in vector store. Please upload a PDF to data/docs/."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = (
        "You are a Senior Risk Officer at Deloitte Financial Advisory.\n"
        "Answer the user's question using only the provided context from bank credit risk documents.\n"
        "If the answer cannot be determined from context, state clearly that information is unavailable.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query)


if __name__ == "__main__":
    vstore = build_or_load_vectorstore()
    if vstore:
        query = "What is model risk and why must models be validated?"
        answer = query_rag_system(query, vstore)
        print("\n--- RAG QUERY RESPONSE ---")
        print(answer)
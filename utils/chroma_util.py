from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os


def prepare_transcript_for_indexing(processed_transcript, api_key, chunk_size=200, chunk_overlap=20):
    """
    Chunk transcript and setup embedding model in one step.
    
    :param processed_transcript: The processed transcript text
    :param api_key: OpenAI API key
    :param chunk_size: Size of each chunk
    :param chunk_overlap: Overlap between chunks
    :return: Tuple of (chunks, embedding_model)
    """
    # Chunk the transcript
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(processed_transcript)
    
    # Setup embedding model
    embedding_model = OpenAIEmbeddings(
        model='text-embedding-3-small', 
        api_key=api_key
    )
    
    return chunks, embedding_model


def get_or_create_chroma_index(chunks=None, embedding_model=None, use_existing=False, 
                                collection_name="youtube_transcripts", persist_directory="./chroma_db"):
    """
    Get existing ChromaDB index or create a new one.
    
    :param chunks: List of text chunks (required if use_existing=False)
    :param embedding_model: The embedding model to use
    :param use_existing: Whether to load existing index or create new
    :param collection_name: Name for the Chroma collection
    :param persist_directory: Directory to store the database
    :return: Chroma index
    """
    if use_existing:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
    else:
        return Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )


def retrieve_and_format_context(query, vector_index, k=7):
    """
    Retrieve relevant context from the vector index and format it.

    :param query: The user's query string
    :param vector_index: The vector index containing the embedded documents
    :param k: The number of most relevant documents to retrieve
    :return: Formatted string of relevant context
    """
    docs = vector_index.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


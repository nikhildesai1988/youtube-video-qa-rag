from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    """
    Chunk the transcript into smaller pieces for embedding.
    
    :param processed_transcript: The processed transcript text
    :param chunk_size: Size of each chunk
    :param chunk_overlap: Overlap between chunks
    :return: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def setup_embedding_model(api_key):
    """
    Set up the OpenAI embedding model.
    
    :param api_key: OpenAI API key
    :return: OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(
        model='text-embedding-3-small', 
        api_key=api_key
    )


def create_chroma_index(chunks, embedding_model, collection_name="youtube_transcripts", persist_directory="./chroma_db"):
    """
    Create a Chroma index from text chunks using the specified embedding model.
    Persists to disk by default.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :param collection_name: Name for the Chroma collection
    :param persist_directory: Directory to store the database
    :return: Chroma index
    """
    return Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )


def load_chroma_index(embedding_model, collection_name="youtube_transcripts", persist_directory="./chroma_db"):
    """
    Load an existing Chroma index from disk.
    
    :param embedding_model: The embedding model to use
    :param collection_name: Name of the Chroma collection
    :param persist_directory: Directory where the database is stored
    :return: Chroma index
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )


def retrieve(query, vector_index, k=7):
    """
    Retrieve relevant context from the vector index based on the user's query.

    Parameters:
        query (str): The user's query string.
        vector_index (Chroma): The vector index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 7).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = vector_index.similarity_search(query, k=k)
    return relevant_context

import re
import os
from youtube_transcript_api import YouTubeTranscriptApi 
from .llm_utils import initialize_openai_llm
from .chroma_util import prepare_transcript_for_indexing, get_or_create_chroma_index
from .chaining_util import create_summary_chain, answer_with_context
from dotenv import load_dotenv

load_dotenv(override=True)


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def getTranscript(url):
    video_id = get_video_id(url)

    ytt_api = YouTubeTranscriptApi()

    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop

    return transcript.to_raw_data()
        

def process_transcript(transcript):
    print(transcript)
    txt = ""
    # 'i' is now a dictionary like {'text': 'Hello', 'start': 0.5}
    for i in transcript:
        try:
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except (KeyError, TypeError):
            pass
    return txt


def summarize_video(video_url):
    """
    Summarize a YouTube video using its transcript.
    
    :param video_url: YouTube video URL
    :return: Summary text
    """
    global fetched_transcript, processed_transcript
    
    if not video_url:
        return "Please provide a valid YouTube URL."
    
    # Fetch and process transcript
    fetched_transcript = getTranscript(video_url)
    processed_transcript = process_transcript(fetched_transcript)
    print(f"Processed Transcript: {processed_transcript}")

    if not processed_transcript:
        return "No transcript available for this video."
    
    # Setup LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = initialize_openai_llm("gpt-4o", api_key)

    # Create summary chain and generate summary
    summary_chain = create_summary_chain(llm)
    result = summary_chain.invoke({"transcript": processed_transcript})
    
    return result.content


def answer_question(video_url, user_question, use_existing_index=False):
    """
    Answer questions about a YouTube video using RAG.
    
    :param video_url: YouTube video URL
    :param user_question: User's question about the video
    :param use_existing_index: Whether to use existing ChromaDB index
    :return: Answer text
    """
    global fetched_transcript, processed_transcript

    # Fetch transcript if needed
    if not processed_transcript:
        if not video_url:
            return "Please provide a valid YouTube URL."
        fetched_transcript = getTranscript(video_url)
        processed_transcript = process_transcript(fetched_transcript)

    if not processed_transcript or not user_question:
        return "Please provide a valid question and ensure the transcript has been fetched."

    # Setup LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = initialize_openai_llm("gpt-4o", api_key)

    # Prepare chunks and embeddings
    chunks, embedding_model = prepare_transcript_for_indexing(processed_transcript, api_key)
    
    # Get or create ChromaDB index
    chroma_index = get_or_create_chroma_index(
        chunks=chunks, 
        embedding_model=embedding_model, 
        use_existing=use_existing_index
    )

    # Generate answer with context
    answer = answer_with_context(user_question, chroma_index, llm)
    return answer
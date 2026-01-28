
import os
from openai import OpenAI, api_key 

from youtube_utils import getTranscript, process_transcript
from llm_utils import setup_credentials, initialize_openai_llm
from chroma_util import chunk_transcript, setup_embedding_model, create_chroma_index, load_chroma_index
from chaining_util import (
    create_summary_prompt, 
    create_summary_chain, 
    create_qa_prompt_template, 
    create_qa_chain, 
    generate_answer
)

from dotenv import load_dotenv

load_dotenv(override=True)


def summarize_video(video_url):
    """
    Title: Summarize Video (OpenAI Edition)
    """
    global fetched_transcript, processed_transcript
    
    if video_url:
        # Fetch and preprocess transcript (Logic remains the same)
        fetched_transcript = getTranscript(video_url)
        #print(f"Fetched Transcript: {fetched_transcript}")
        processed_transcript = process_transcript(fetched_transcript)
        print(f"Processed Transcript: {processed_transcript}")
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Step 1: Set up OpenAI credentials 
        # (Using the refactored function from earlier)
        client= setup_credentials()

        # Step 2: Initialize OpenAI LLM
        # Parameters should now include 'max_tokens' instead of 'max_new_tokens'
        api_key = os.getenv("OPENAI_API_KEY")
        llm = initialize_openai_llm("gpt-4o", api_key)

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        # If you used LLMChain, keep .run(). If you used LCEL, use .invoke()
        summary = summary_chain.run({"transcript": processed_transcript})
        
        return summary
    else:
        return "No transcript available for this video."


def answer_question(video_url, user_question, use_existing_index=False):
    """
    Title: Answer User's Question (OpenAI + ChromaDB Edition)
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            fetched_transcript = getTranscript(video_url)
            processed_transcript = process_transcript(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up OpenAI credentials
        client = setup_credentials()
        api_key = os.getenv("OPENAI_API_KEY")

        # Step 3: Initialize OpenAI LLM for Q&A
        llm = initialize_openai_llm("gpt-4o", api_key)

        # Step 4: Create or load ChromaDB index with OpenAI Embeddings
        embedding_model = setup_embedding_model(api_key)
        
        if use_existing_index:
            chroma_index = load_chroma_index(embedding_model)
        else:
            chroma_index = create_chroma_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer
        answer = generate_answer(user_question, chroma_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
    



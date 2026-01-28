from openai import OpenAI
from langchain_openai import ChatOpenAI
import os


def initialize_openai_llm(model_id, api_key):
    """
    Create and return an instance of ChatOpenAI with the specified configuration.
    """
    return ChatOpenAI(
        model=model_id,           # e.g., "gpt-4o"
        api_key=api_key,          # Your OpenAI API key
        temperature=0.7)
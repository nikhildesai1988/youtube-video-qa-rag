from langchain_core.prompts import PromptTemplate
from .chroma_util import retrieve_and_format_context


def create_summary_chain(llm, verbose=True):
    """
    Create a complete LCEL chain for summarizing transcripts with built-in prompt.
    
    :param llm: Language model instance
    :param verbose: Whether to enable verbose output (unused in LCEL)
    :return: LCEL chain ready for summarization
    """
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = PromptTemplate(input_variables=["transcript"], template=template)
    return prompt | llm


def answer_with_context(question, vector_index, llm, k=7, verbose=True):
    """`
    Retrieve context and generate answer in one step.
    
    :param question: The user's question
    :param vector_index: The vector index containing the embedded documents
    :param llm: Language model instance
    :param k: Number of relevant documents to retrieve
    :param verbose: Whether to enable verbose output (unused in LCEL)
    :return: Generated answer string
    """
    # Retrieve relevant context
    context = retrieve_and_format_context(question, vector_index, k=k)
    
    # Create Q&A prompt and chain using LCEL
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=qa_template)
    qa_chain = prompt | llm
    
    # Generate answer using LCEL invoke
    result = qa_chain.invoke({"context": context, "question": question})
    return result.content


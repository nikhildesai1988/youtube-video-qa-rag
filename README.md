# YouTube Video Q&A

An AI-powered application that allows you to summarize YouTube videos and ask questions about their content using RAG (Retrieval-Augmented Generation) with OpenAI and ChromaDB.

## Features

- 📝 **Video Summarization**: Automatically generate concise summaries of YouTube video transcripts
- 💬 **Interactive Q&A**: Ask questions about video content and get AI-powered answers
- 🗄️ **Persistent Storage**: Uses ChromaDB for efficient vector storage and retrieval
- 🚀 **Modern UI**: Clean Gradio interface for easy interaction
- 🔍 **RAG Pipeline**: Implements retrieval-augmented generation for accurate, context-aware responses

## Architecture

The project is organized into modular components:

- **`utils/`**: All core functionality and utility modules
  - **`youtube_utils.py`**: Main YouTube processing module
    - `get_video_id()` - Extract video ID from URL
    - `getTranscript()` - Fetch YouTube transcript
    - `process_transcript()` - Clean and format transcript text
    - `summarize_video()` - Generate video summaries
    - `answer_question()` - Answer questions using RAG
  - **`chroma_util.py`**: ChromaDB and embedding utilities
    - `prepare_transcript_for_indexing()` - Chunks text and sets up embeddings
    - `get_or_create_chroma_index()` - Unified index management
    - `retrieve_and_format_context()` - Semantic search and formatting
  - **`chaining_util.py`**: LangChain chains (simplified, all-in-one functions)
    - `create_summary_chain()` - Complete summarization chain
    - `answer_with_context()` - RAG pipeline (retrieve + generate)
  - **`llm_utils.py`**: OpenAI LLM initialization and configuration
- **`gradio_ui.py`**: Web interface using Gradio
- **`main.py`**: Application entry point

## Prerequisites

- Python 3.13+
- OpenAI API key
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd youtube_video_QA

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <your-repo-url>
cd youtube_video_QA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Application

```bash
# Using UV
uv run main.py

# Using Python
python main.py
```

The application will launch at `http://localhost:7860`

### Using the Interface

1. **Enter YouTube URL**: Paste the URL of any YouTube video with available transcripts
2. **Summarize**: Click "Summarize Video" to get a concise summary
3. **Ask Questions**: Type your question and click "Ask a Question" to get answers based on the video content

### Programmatic Usage

```python
from utils.youtube_utils import summarize_video, answer_question

# Summarize a video
video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
summary = summarize_video(video_url)
print(summary)

# Ask a question
question = "What are the main points discussed?"
answer = answer_question(video_url, question, use_existing_index=False)
print(answer)

# Reuse existing ChromaDB index (faster for follow-up questions)
answer = answer_question(video_url, question, use_existing_index=True)
```

## How It Works

1. **Transcript Extraction**: Fetches the video transcript using `youtube-transcript-api`
2. **Text Processing**: `prepare_transcript_for_indexing()` chunks the transcript and initializes embeddings in one step
3. **Vector Storage**: `get_or_create_chroma_index()` stores embeddings in ChromaDB (persisted to `./chroma_db/`)
4. **Question Answering**: `answer_with_context()` performs the complete RAG pipeline:
   - Retrieves relevant chunks using similarity search
   - Formats context for the LLM
   - Generates answer using GPT-4o with retrieved context
5. **Summarization**: `create_summary_chain()` generates concise summaries directly from the full transcript

## Technology Stack

- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **UI**: Gradio
- **Package Manager**: UV

## Project Structure

```
youtube_video_QA/
├── main.py                 # Application entry point
├── gradio_ui.py           # Gradio web interface
├── utils/                 # Core functionality and utility modules
│   ├── __init__.py        # Package initializer
│   ├── youtube_utils.py   # YouTube transcript handling, summarization, Q&A
│   ├── chroma_util.py     # ChromaDB utilities
│   ├── chaining_util.py   # LangChain chains and prompts
│   └── llm_utils.py       # LLM initialization
├── pyproject.toml         # Project dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Dependencies

Key dependencies include:
- `langchain==0.2.6`
- `langchain-openai>=0.1.25`
- `langchain-chroma>=0.2.2`
- `chromadb>=0.6.3`
- `gradio>=6.4.0`
- `youtube-transcript-api==1.2.1`
- `openai`

See [pyproject.toml](pyproject.toml) for complete dependency list.

## Features in Detail

### Persistent Vector Storage

ChromaDB stores embeddings in the `./chroma_db/` directory, allowing you to:
- Reuse embeddings across sessions
- Avoid re-embedding the same content
- Save API costs and processing time

### Modular Architecture

The codebase is designed for maintainability and simplicity:
- **Consolidated functions**: Related operations grouped together (e.g., `prepare_transcript_for_indexing()` does chunking + embedding setup)
- **Separation of concerns**: Each module has a specific purpose
- **Easy testing**: Modular functions are easier to test and understand
- **Flexible**: Swap out components (e.g., different LLMs or vector stores) easily

## Troubleshooting

### Import Errors
If you see import resolution errors in your IDE, ensure you're using the correct Python interpreter (the UV virtual environment or your project's venv).

### API Key Issues
Make sure your `.env` file contains a valid OpenAI API key:
```env
OPENAI_API_KEY=sk-...
```

### ChromaDB Persistence Issues
If you want to start fresh, delete the `chroma_db/` directory:
```bash
rm -rf chroma_db/
```

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Uses [OpenAI](https://openai.com/) models
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- UI built with [Gradio](https://gradio.app/)

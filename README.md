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

- **`ybot.py`**: Main orchestration layer with `summarize_video()` and `answer_question()` functions
- **`chroma_util.py`**: ChromaDB and embedding utilities (chunking, indexing, retrieval)
- **`chaining_util.py`**: LangChain prompt templates and chain creation
- **`llm_utils.py`**: OpenAI LLM initialization and configuration
- **`youtube_utils.py`**: YouTube transcript fetching and processing
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
from ybot import summarize_video, answer_question

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
2. **Text Chunking**: Splits the transcript into manageable chunks using `RecursiveCharacterTextSplitter`
3. **Embedding**: Converts chunks to vector embeddings using OpenAI's `text-embedding-3-small` model
4. **Vector Storage**: Stores embeddings in ChromaDB for persistent, efficient retrieval
5. **Retrieval**: Finds relevant chunks using similarity search
6. **Generation**: Uses GPT-4o to generate answers based on retrieved context

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
├── ybot.py                 # Core orchestration logic
├── chroma_util.py         # ChromaDB utilities
├── chaining_util.py       # LangChain chains and prompts
├── llm_utils.py           # LLM initialization
├── youtube_utils.py       # YouTube transcript handling
├── gradio_ui.py           # Gradio web interface
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

The codebase is designed for maintainability:
- **Separation of concerns**: Each module has a specific purpose
- **Easy testing**: Modular functions are easier to test
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Uses [OpenAI](https://openai.com/) models
- Vector storage powered by [ChromaDB](https://www.trychroma.com/)
- UI built with [Gradio](https://gradio.app/)

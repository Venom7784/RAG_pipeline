# RAG Pipeline

A Chainlit-based Retrieval-Augmented Generation (RAG) app for answering questions about PDF documents using LangChain, ChromaDB, and Groq.

## Features

- PDF document loading and processing
- Text chunking and embedding generation
- Vector storage with ChromaDB
- Question answering with Groq LLM
- Chainlit chat interface
- PDF upload directly from the Chainlit attachment (paperclip) button
- Optional debug commands: `/debug on`, `/debug off`, `/debug status`

## Local Development

1. Create a virtual environment:
   ```bash
   uv venv
   ```

2. Activate it:
   ```bash
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key as `GROQ_API_KEY`
   - Do not commit your real key to the repository

5. Run the app:
   ```bash
   uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8000
   ```

6. Open `http://localhost:8000`
7. Upload PDFs using the attachment (paperclip) button in Chainlit, then ask questions.

If `uv` hits a Windows cache permission issue, use:

```bash
$env:UV_CACHE_DIR=".uv-cache"
uv sync
uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8000
```

## Environment Variables

Required:
- `GROQ_API_KEY`

Common:
- `PDF_DIRECTORY`
- `PERSIST_DIRECTORY`
- `COLLECTION_NAME`
- `EMBEDDING_MODEL_NAME`
- `GROQ_MODEL_NAME`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TEMPERATURE`
- `MAX_TOKENS`
- `RETRIEVAL_RESULTS`
- `SIMILARITY_THRESHOLD`

## Notes

- If the local PDF directory is empty, the app still starts and waits for PDF uploads via attachment.
- By default for local development, the app uses the repo's `data/` folder.
- Uploaded PDFs persist on disk unless you delete the configured data directories.
- Use `/debug on` to include tool and source details in responses.

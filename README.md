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

## Windows Desktop App

This repo can also be packaged as a Windows desktop app without removing the existing Chainlit app.

How it works:
- `desktop_launcher.py` starts the Chainlit server locally
- `pywebview` opens the app inside a native desktop window
- desktop mode stores PDFs and the Chroma vector store under `%LOCALAPPDATA%\RAGPipeline\data` by default

Run the desktop shell from source:
```bash
uv run python desktop_launcher.py
```

Build the Windows executable:
```powershell
.venv\Scripts\python -m pip install pyinstaller
.\build_windows.ps1
```

The packaged app will be created under `dist\RAGPipelineDesktop\`.

Notes:
- `GROQ_API_KEY` is still required, usually via `.env`
- the existing `chainlit run chainlit_app.py ...` flow continues to work unchanged
- the first desktop launch may take longer while models and caches initialize
- in desktop mode, uploaded PDFs and vectors are persisted under `%LOCALAPPDATA%\RAGPipeline\data`

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

## Railway Deployment

This project is set up to run on Railway as a Chainlit app.

Start command:
```bash
python -m chainlit run chainlit_app.py --host 0.0.0.0 --port $PORT
```

A `Procfile` is included for Railway.

Set these Railway environment variables:
- `GROQ_API_KEY`
- `DATA_ROOT=/data`
- `PDF_DIRECTORY=/data/pdf`
- `PERSIST_DIRECTORY=/data/vector_store`

Recommended Railway setup:
1. Set the root directory to the repo root.
2. Let Railway detect Python.
3. Add your `GROQ_API_KEY` in Railway Variables.
4. Attach a Railway Volume mounted at `/data`.
5. Redeploy after adding the variables.

## Notes

- For Railway with a mounted volume, keep your PDFs in `/data/pdf`.
- The Chroma vector store should persist under `/data/vector_store`.
- On a fresh Railway volume, the app creates `/data/pdf` and `/data/vector_store` automatically.
- If `/data/pdf` is empty, the app still starts and waits for PDF uploads via attachment.
- For local development without Railway vars, the app falls back to the repo's `data/` folder.
- Uploaded PDFs persist on disk unless you delete the configured data directories.
- Use `/debug on` to include tool and source details in responses.

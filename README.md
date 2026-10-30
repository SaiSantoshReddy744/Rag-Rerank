# RagEval - Document Question Answering with Neural Reranking

Upload a PDF → text is extracted, chunked, embedded (`BAAI/bge-small-en-v1.5`) and stored in Qdrant →
a question is answered by **vector retrieval (top 20) -> cross-encoder reranking (`BAAI/bge-reranker-base`) ->
top passages sent to OpenAI's Responses API**. One FastAPI app serves both the API and the web UI.

## 1. Requirements
- Windows 10/11, **Python 3.11+** (3.12 is fine), internet access
- An OpenAI API key with API billing/credits enabled: https://platform.openai.com/api-keys
- No Docker / Qdrant server needed (an embedded on-disk Qdrant is used when `QDRANT_URL` is empty)

## 2. Install (PowerShell, in the project folder)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

In `.env`, set `OPENAI_API_KEY=` to your key and choose an available model for `OPENAI_MODEL=`.
**Do not paste your key anywhere else or commit `.env`.** ChatGPT subscriptions and API billing are managed separately.

## 3. Choose an OpenAI model your API project can use
```powershell
python -m app.tools.list_models
```
This calls OpenAI's models endpoint and lists model IDs visible to your key. Set one text model ID in `OPENAI_MODEL=`.
The app sends prompts through the Responses API.

## 4. Run
```powershell
python run.py
```
Opens http://127.0.0.1:8000 . (Equivalent: `python -m uvicorn app.main:app`.)

### Run the Streamlit interface instead
```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```
Streamlit opens its local URL (usually http://localhost:8501). The FastAPI UI remains available with `python run.py`.
For Streamlit Community Cloud, deploy `streamlit_app.py` from your GitHub repository and add the API settings under the app's Secrets settings. Never commit `.env` or API keys.

## 5. Use it
1. Click the drop area and choose a PDF (text-based PDFs only; scanned images have no text and are rejected).
   The **first** upload downloads the embedding model (~130 MB) — wait for "indexed".
2. Type a question, press **Ask Question**. The **first** question downloads the reranker (~1.1 GB).
3. Read the answer and source passages. Open **Retrieval Results** to compare the order before and after reranking
   (green border = passage sent to OpenAI). The **Evaluation** tab shows stored results from `eval/results/`.

## 6. Configuration (`.env`)
| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_MODEL` | required |
| `QDRANT_URL` | empty = embedded DB in `data/qdrant`; set to use a Qdrant server |
| `RE_RANK_ENABLED` | `true` by default |
| `RERANK_CANDIDATE_K` | candidates fetched before reranking (default 20) |
| `MAX_UPLOAD_SIZE_BYTES` | upload limit (default 52428800, 50 MiB) |
| `EMBEDDING_BATCH_SIZE` | chunks embedded per batch (default 64; lower it if memory is tight) |

`.env` is read from the project root regardless of the working directory, and is git-ignored.

## 7. Error messages you may see
- *OPENAI_API_KEY is not set* / *OPENAI_MODEL is not set* -> edit `.env`, restart.
- *OpenAI rejected the API key* -> check the key and project access.
- *Model 'x' was not found* -> run `python -m app.tools.list_models` and choose an available model.
- *quota or rate limit exceeded* (429) / *temporarily overloaded* (503) → retried up to 3 times with backoff, then reported.
- *Neural reranking failed: …* → reranker could not load/run; it is never silently skipped.

## 8. Tests
```powershell
python -m pytest -q
```
Tests mock the OpenAI API and vector DB; they do not call the network.

## 9. Honest notes
- Evaluation numbers in the UI are read from `eval/results/*.json` (earlier real runs on the repository's
  evaluation PDF). They are not recomputed for your uploads. See `eval/README.md`.
- Retrieval/rerank scores shown in the UI are the actual vector similarity and cross-encoder outputs.
- Scanned (image-only) PDFs are not supported (no OCR).
- If the `cl100k_base` tokenizer vocabulary is not already cached, chunking and prompt limits use an offline approximation instead of trying to download it during startup.

## Project layout
`app/` FastAPI backend (`api/`, `core/`, `ingestion/`, `static/index.html` UI) · `eval/`, `benchmarks/` offline evaluation · `tests/`.
Original design notes: see `ROADMAP.md`, `CHANGELOG.md`.

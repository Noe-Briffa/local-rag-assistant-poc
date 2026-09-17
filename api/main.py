# Lancer avec: python -m uvicorn api.main:app --reload --port 8000

import os
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, rag_local
from .indexer import build_chroma_db_from_records, make_records_for_chroma

# --- Constantes ---
WEB_DIR = config.WEB_DIR

app = FastAPI(title="Local RAG Assistant", version="1.0")

# --- Fichiers statiques ---
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir la SPA si présente dans WEB_DIR, sinon un message minimal."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Local RAG Assistant</h1><p>Placez l'UI dans ./web/index.html</p>")


@app.get("/health")
async def health():
    try:
        # si /health du serveur LLM n'existe pas, on peut fallback sur /version
        with httpx.Client(timeout=3.0) as c:
            r = c.get(f"{config.LLAMA_SERVER_URL}/health")
            if r.status_code == 200:
                llm_ok = True
            else:
                # fallback léger (certaines images exposent /version)
                r2 = c.get(f"{config.LLAMA_SERVER_URL}/version")
                llm_ok = r2.status_code == 200
    except httpx.HTTPError:
        llm_ok = False

    chroma_ok = os.path.isdir(config.CHROMA_DIR)
    ok = llm_ok and chroma_ok
    return {"ok": ok, "llm_ok": llm_ok, "chroma_ok": chroma_ok}



@app.post("/ingest")
async def ingest():
    """(Re)construit la base Chroma à partir du PDF d'entrée."""
    t0 = time.time()
    if not os.path.isfile(config.PDF_IN):
        return JSONResponse({"ok": False, "error": "Document source introuvable."}, status_code=400)

    records = make_records_for_chroma(
        config.PDF_IN,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        source_name=os.path.basename(config.PDF_IN),
    )
    build_chroma_db_from_records(
        records,
        persist_dir=config.CHROMA_DIR,
        collection_name=config.CHROMA_COLLECTION,
        rebuild=True,
    )
    dt = round((time.time() - t0) * 1000)
    return {"ok": True, "count": len(records), "ms": dt}


@app.post("/chat")
async def chat(payload: dict):
    q = (payload or {}).get("question", "").strip()
    if not q:
        return JSONResponse({"error": "Champ 'question' requis."}, status_code=400)
    t0 = time.time()
    try:
        res = rag_local.rag_answer(q, k=config.TOP_K)
        res["latency_ms"] = round((time.time() - t0) * 1000)
        return res
    except Exception:  # noqa: BLE001 - l'API ne doit pas exposer les détails internes
        return JSONResponse({"error": "Erreur interne pendant la génération."}, status_code=500)


# --- Hook optionnel: ingestion auto au démarrage ---
@app.on_event("startup")
async def _maybe_ingest_on_start():
    if getattr(config, "DO_INGEST", False):
        try:
            await ingest()  # rebuild
        except Exception:  # noqa: BLE001 - échec non bloquant au démarrage
            return

    # 2) WARMUP LLM : déclencher le chargement du modèle AVANT la 1re question
    try:
        with httpx.Client(timeout=300.0) as client:
            client.post(
                f"{config.LLAMA_SERVER_URL}/completion",
                json={"prompt": "ok", "n_predict": 1}
            )
    except httpx.HTTPError:
        return

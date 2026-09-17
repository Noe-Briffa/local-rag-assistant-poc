import os
import re
from functools import lru_cache

import chromadb
import numpy as np
import pymupdf
import torch
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

from . import config

# global
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_encoder():
    return SentenceTransformer(config.EMBEDDING_MODEL, device=_DEVICE)


def extract_pages_from_pdf(pdf_path: str):
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):  # numérotation à partir de 1
            text = page.get_text("text") or ""
            # nettoyage simple des pieds de page du type "120/191"
            text = re.sub(r"\b\d{1,4}\s*/\s*\d{1,4}\b", "", text)
            # normalisation espaces
            text = re.sub(r"[ \t]+", " ", text).strip()
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: config.CHUNK_SIZE, overlap: config.CHUNK_OVERLAP):
    """
    Découpe le texte d'entrée en chunk
    :param text: Texte d'entrée
    :param chunk_size: Taille (en caractère) d'un chunk
    :param overlap: Taille (en caractère) de l'overlap (texte partagé entre 2 chunks)
    :return:
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size doit être positif et overlap inférieur à chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def make_records_for_chroma(pdf_path: str, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, source_name=None):
    pages = extract_pages_from_pdf(pdf_path)
    records = []
    source = source_name or os.path.basename(pdf_path)
    for p in pages:
        chunks = chunk_text(p["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, ck in enumerate(chunks):
            if ck:
                records.append({
                    "text": ck,
                    "source": source,
                    "page": p["page"],
                    "chunk_index": idx,
                })
    return records


def build_chroma_db_from_records(
        records,
        persist_dir=config.CHROMA_DIR,
        collection_name=config.CHROMA_COLLECTION,
        model_name=config.EMBEDDING_MODEL,
        rebuild=True,
):
    """
    Construit (ou remplace) une collection Chroma avec embeddings
    :param records: Liste de Dict avec le chunk et ses attributs
    :param persist_dir: Dossier persistant Chroma
    :param collection_name: Nom de la collection Chroma
    :param model_name: Nom du modèle utilisé pour l'embedding
    :param rebuild: Est ce que la BDD doit être reconstruite
    :return: La collection Chroma
    """
    os.makedirs(persist_dir, exist_ok=True)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            pass
        col = client.create_collection(name=collection_name, embedding_function=embedding_fn)
    else:
        try:
            col = client.get_collection(name=collection_name, embedding_function=embedding_fn)
        except NotFoundError:
            col = client.create_collection(name=collection_name, embedding_function=embedding_fn)

    docs = [r["text"] for r in records]
    metas = [{"source": r["source"], "page": r["page"], "chunk_index": r["chunk_index"]} for r in records]
    ids = [f'{r["source"]}:{r["page"]}:{r["chunk_index"]}' for r in records]

    col.add(documents=docs, metadatas=metas, ids=ids)
    return col


def retrieve_topk(
    query: str,
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
    model_name: str = config.EMBEDDING_MODEL,
    k: int = config.TOP_K,
    candidate_pool=config.CANDIDATE_POOL,
    lambda_diversity=config.MMR_DIVERSITY,  # 0 = uniquement pertinence, 1 = uniquement diversité
):
    """
    Recharge la collection persistée et renvoie les top-k passages proches de la requête
    :param query: Requête (question) utilisateur
    :param persist_dir: Dossier de sauvegarde Chroma
    :param collection_name: Nom de la collection Chroma
    :param model_name: Modèle d'embedding
    :param k: Nombre de chunks à sélectionner (le splus proches de la requête)
    :param candidate_pool: Nombre de chunks candidats
    :param lambda_diversity: Équilibre entre pertinence et diversité
    :return: Liste de k chunks correspondants à la requête
    """
    # 0) Fonction MMR
    def mmr(query_emb, doc_embs, top_k=3, diversity=0.5):
        """
        Maximal Marginal Relevance (MMR)
        :param query_emb: vecteur (dim,)
        :param doc_embs: np.array (N, dim)
        :param top_k: Nombre de passages à retourner
        :param diversity: Équilibre pertinence/diversité (0=seulement pertinence)
        :return:
        """
        from numpy import dot
        from numpy.linalg import norm

        def cosine(a, b):
            return dot(a, b) / (norm(a) * norm(b) + 1e-10)

        n = doc_embs.shape[0]
        if n <= top_k:
            return list(range(n))

        selected = []
        candidates = list(range(n))

        # Similarité requête-doc
        sim_to_query = [cosine(query_emb, d) for d in doc_embs]

        # Premier = plus proche de la requête
        first = int(np.argmax(sim_to_query))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < top_k and candidates:
            mmr_scores = []
            for c in candidates:
                sim1 = sim_to_query[c]
                sim2 = max(cosine(doc_embs[c], doc_embs[s]) for s in selected)
                score = diversity * sim1 - (1 - diversity) * sim2
                mmr_scores.append((score, c))
            mmr_scores.sort(reverse=True, key=lambda x: x[0])
            best = mmr_scores[0][1]
            selected.append(best)
            candidates.remove(best)

        return selected

    # 1) Charger la collection et prendre un pool plus grand
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
    col = client.get_collection(name=collection_name, embedding_function=embedding_fn)
    res = col.query(
        query_texts=[query],
        n_results=max(candidate_pool, k),
        include=["documents", "metadatas", "distances", "embeddings"]  # <-- important
    )
    if not res["ids"] or len(res["ids"][0]) == 0:
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    ids = res["ids"][0]
    embs = np.array(res["embeddings"][0], dtype=np.float32)

    # 2) Embedding de la requête
    q_emb = _get_encoder().encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    # 3) Sélection MMR (pertinence + diversité)
    picked = mmr(q_emb, embs, top_k=k, diversity=lambda_diversity)

    # 4) Retour au même format que ton code actuel
    hits = []
    for i in picked:
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": res["distances"][0][i],
        })
    return hits

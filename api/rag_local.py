import re

import httpx

from . import config
from .indexer import retrieve_topk

LLAMA_SERVER_URL = config.LLAMA_SERVER_URL
MODEL_PROMPT_TPL = """Tu es un assistant RAG local. Réponds en français.

[CONTEXTE]
{context}

[INSTRUCTIONS]
- Réponds UNIQUEMENT avec les informations du CONTEXTE.
- Donne une réponse directe et concise (pas de plan, pas "Solution 1", pas "Instruction", pas "Réponse :", etc).
- Réponds de manière concise mais clair et pertinente.
- N'imprime PAS de balises <|user|>, <|assistant|>, [REPONSE], ni d'en-têtes ou gabarits.

[QUESTION]
{question}
"""


def format_context(hits: list[dict], max_chars_total=config.MAX_CHARS_TOTAL, per_snippet=config.PER_SNIPPET) -> str:
    """
    Cette fonction prend les résultats de la recherche vectorielle (les hits renvoyés par Chroma) et les transforme en
    un contexte compact à injecter dans le prompt du LLM.
    :param hits: List des passages retenus par notre fonction de retrieving
    :param max_chars_total: Total de contexte à ne pas dépasser
    :param per_snippet: Nombre de caractères gardés par chunk (pour ne pas surcharger le LLM)
    :return: Le contexte global à intégrer au prompt
    """
    parts, used = [], 0
    for h in hits:
        txt = (h["text"] or "").strip()
        if per_snippet == -1: snippet = txt.replace("\n", " ")
        else: snippet = txt[:per_snippet].replace("\n", " ")
        meta = h.get("metadata", {})
        src = meta.get("source", "doc")
        page = meta.get("page", "?")
        block = f"(Source: {src} p.{page}) {snippet}"
        if used + len(block) > max_chars_total:
            break
        parts.append(block)
        used += len(block)
    return "\n- ".join(parts) if parts else "Aucun extrait pertinent."


def call_llama_server(prompt: str, max_tokens=config.N_PREDICT) -> str:
    """
    Cette fonction envoie le prompt préparé au serveur llama.cpp (qui fait tourner ton LLM localement sur GPU),
    et récupère la réponse.
    :param prompt: Le prompt à passer au LLM
    :param max_tokens: Nombre de token max pour la réponse du LLM
    :return: Réponse du LLM
    """
    payload = {
        "prompt": prompt,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "repeat_penalty": config.REPEAT_PENALTY,
        "n_predict": max_tokens,
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
        r.raise_for_status()
        data = r.json()

    txt = (data.get("content") or "").strip()

    # enlever les balises (si présentes)
    for tok in config.STOP_TOKENS:
        if tok in txt:
            txt = txt.replace(tok, "").strip()

    return txt


def rag_answer(query: str, retriever=lambda **kw: retrieve_topk(**kw), k=config.TOP_K) -> dict:
    """
    Fonction orchestratrice du RAG : elle relie le retrieval et la génération pour donner une réponse finale.
    :param query: Requête utilisateur
    :param retriever:
    :param k: Nombre de chunks à pull
    :return: Un json contenant la réponse du LLM
    """
    # fonction pour compter les mots
    def _count_words(txt: str) -> int:
        if not txt:
            return 0
        # Compte robuste (lettres/chiffres, FR-friendly)
        return len(re.findall(r"[\w’'-]+", txt, flags=re.UNICODE))

    # 1) retrieve
    hits = retriever(query=query, k=k)

    # 2) prompt
    ctx = format_context(hits)
    prompt = MODEL_PROMPT_TPL.format(context=ctx, question=query)

    # 3) generate
    answer = call_llama_server(prompt, max_tokens=config.N_PREDICT)

    # 4) post-process (sources)
    sources = []
    for h in hits:
        m = h.get("metadata", {})
        src = m.get("source")
        page = m.get("page")
        if src:
            label = f"{src} p.{page}" if page else src
            if label not in sources:
                sources.append(label)

    return {
        "answer": answer,
        "sources": sources,
        "num_ctx_chars": len(ctx),
        "num_hits": len(hits),
        "num_ctx_words": _count_words(ctx),
    }

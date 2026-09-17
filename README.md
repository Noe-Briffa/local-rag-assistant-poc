# Local RAG Assistant

POC de question-réponse sur un document local. L’application extrait le texte d’un PDF, construit un index vectoriel ChromaDB, sélectionne des passages pertinents puis demande à un modèle GGUF servi par `llama.cpp` de produire une réponse sourcée.

Le dépôt ne contient ni modèle, ni document, ni index préconstruit. Ces éléments restent locaux.

## Architecture

```text
Navigateur
   │
   ├── interface HTML statique
   │
   └── FastAPI ── ChromaDB ── embeddings sentence-transformers
           │
           └── llama-server ── modèle GGUF local
```

- `api/` : API, ingestion, recherche vectorielle et génération.
- `web/` : interface sans étape de build.
- `ops/` : proxy nginx utilisé par Docker Compose.
- `llama.cpp/` : sous-module épinglé pour le serveur local.
- `Models/`, `data_in/`, `chroma_db/` : données locales ignorées par Git.

## Prérequis

- Git avec prise en charge des sous-modules.
- Python 3.10.
- [`uv`](https://docs.astral.sh/uv/) ou `pip`.
- Un modèle compatible GGUF, obtenu séparément et placé dans `Models/`.
- Un PDF personnel ou librement utilisable, placé dans `data_in/`.
- Pour Docker : Docker Compose, GPU NVIDIA et runtime NVIDIA compatibles avec l’image CUDA utilisée.

## Installation native sous Windows

```powershell
git clone --recurse-submodules <URL_DU_DEPOT>
cd local-rag-assistant-poc
Copy-Item .env.example .env
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Renommer ou configurer les fichiers locaux dans `.env` :

```dotenv
LLAMA_MODEL=model.gguf
PDF_IN=data_in/document.pdf
```

Compiler `llama.cpp` selon sa documentation officielle, puis démarrer le serveur :

```powershell
.\start_llama.bat
```

Dans un second terminal :

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Ouvrir <http://127.0.0.1:8000>. L’index peut être construit avec le bouton de l’interface ou avec :

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ingest
```

## Docker Compose

Après avoir créé `.env`, `Models/model.gguf` et `data_in/document.pdf` :

```powershell
docker compose up --build
```

- Interface via nginx : <http://127.0.0.1:8088>
- API directe : <http://127.0.0.1:8000>

La configuration fournie utilise une image `llama.cpp` CUDA. Elle n’a pas de variante CPU implicite.

## API

- `GET /` : interface web.
- `GET /health` : disponibilité de llama.cpp et du répertoire ChromaDB.
- `POST /ingest` : reconstruit l’index depuis `PDF_IN`.
- `POST /chat` avec `{ "question": "..." }` : produit une réponse et ses sources.

## Vérifications

```powershell
uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python.exe -m compileall -q api start_llama.py
uvx ruff check api start_llama.py
```

## Limites et sécurité

- POC local sans authentification ni contrôle d’accès.
- `/ingest` reconstruit la base et ne doit pas être exposé publiquement.
- Les réponses dépendent du document, du modèle et des réglages de génération.
- Le premier usage des embeddings peut télécharger le modèle configuré.
- Les bibliothèques de l’interface sont chargées depuis des CDN externes.
- Aucune licence de réutilisation n’est accordée par défaut.

Ne déployez pas cette configuration directement sur Internet sans ajouter authentification, restrictions réseau et gestion adaptée des données.

import os
import subprocess
import sys

from api import config

exe = getattr(config, "LLAMA_SERVER_EXE", r"llama.cpp\build\bin\Release\llama-server.exe")
model = config.MODEL_PATH
ctx = str(config.CTX_SIZE)
ngl = str(config.N_GPU_LAYERS)
thr = str(config.THREADS)
port = str(config.PORT)

if not os.path.exists(exe):
    print(f"[ERREUR] llama-server introuvable: {exe}", file=sys.stderr)
    sys.exit(1)
if not os.path.exists(model):
    print(f"[ERREUR] Modèle introuvable: {model}", file=sys.stderr)
    sys.exit(1)

cmd = [
    exe,
    "--model", model,
    "--ctx-size", ctx,
    "--n-gpu-layers", ngl,
    "--threads", thr,
    "--port", port,
]

print("=== Démarrage llama-server ===")
print("Commande:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
print(f"URL: {config.LLAMA_SERVER_URL}")

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    pass
except subprocess.CalledProcessError as exc:
    print(f"[ERREUR] llama-server s'est arrêté avec le code {exc.returncode}.", file=sys.stderr)
    sys.exit(exc.returncode)
except OSError:
    print("[ERREUR] Impossible de lancer llama-server. Vérifiez ses dépendances natives.", file=sys.stderr)
    sys.exit(1)

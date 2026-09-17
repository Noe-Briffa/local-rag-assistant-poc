"""Generate the neutral local document used by the end-to-end demo."""

from pathlib import Path

import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "data_in" / "document.pdf"

DEMO_TEXT = """Guide de démonstration

Le projet Aster est un exercice fictif de documentation locale.
Son identifiant de validation unique est ORION-742.
Les réunions de suivi ont lieu le mardi à 10 heures.
Les documents sont conservés uniquement sur la machine de démonstration.

Ce contenu a été rédigé spécialement pour tester l'ingestion, la recherche
vectorielle et la génération de réponses sourcées. Il ne décrit aucune
organisation, aucun produit réel et aucun processus métier existant.
"""


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 523, 770),
        DEMO_TEXT,
        fontsize=12,
        fontname="helv",
        lineheight=1.4,
    )
    document.save(OUTPUT_PATH)
    document.close()
    print("Document de démonstration généré : data_in/document.pdf")


if __name__ == "__main__":
    main()

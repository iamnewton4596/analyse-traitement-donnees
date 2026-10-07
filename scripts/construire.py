"""
Construit les notebooks du dépôt à partir des sources « jupytext » (_sources/).

- _sources/cours/*.py       -> cours/*.ipynb                 (exécutés)
- _sources/exercices/*.py   -> exercices/corriges/*.ipynb    (exécutés, avec solutions)
                            -> exercices/enonces/*.ipynb     (solutions retirées, non exécutés)

Convention : une cellule de code marquée `tags=["solution"]` contient une solution ;
dans l'énoncé elle est remplacée par une cellule vide « # À vous de jouer ».

Auteur : Issa Gueye — Licence MIT
Usage  : python scripts/construire.py [--sans-execution]
"""

import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

RACINE = Path(__file__).resolve().parents[1]
EXECUTER = "--sans-execution" not in sys.argv


def executer(nb, dossier):
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(dossier)}})
    client.execute()
    return nb


def nettoyer_meta(nb):
    nb.metadata.pop("jupytext", None)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["authors"] = [{"name": "Issa Gueye"}]
    return nb


def construire(src, dest, retirer_solutions=False, executer_nb=True):
    nb = jupytext.read(src)
    nb = nettoyer_meta(nb)
    if retirer_solutions:
        cellules = []
        for c in nb.cells:
            if c.cell_type == "code" and "solution" in c.metadata.get("tags", []):
                cellules.append(nbformat.v4.new_code_cell("# À vous de jouer ✍️\n"))
            else:
                c.metadata.pop("tags", None) if c.metadata.get("tags") == [] else None
                cellules.append(c)
        nb.cells = cellules
    dest.parent.mkdir(parents=True, exist_ok=True)
    if executer_nb and EXECUTER:
        executer(nb, dest.parent)
    nbformat.write(nb, dest)
    print(f"  ✔ {dest.relative_to(RACINE)}")


def main():
    cibles = sys.argv[1:]
    cibles = [c for c in cibles if not c.startswith("--")]
    for src in sorted((RACINE / "_sources" / "cours").glob("*.py")):
        if cibles and not any(c in src.name for c in cibles):
            continue
        construire(src, RACINE / "cours" / (src.stem + ".ipynb"))
    for src in sorted((RACINE / "_sources" / "exercices").glob("*.py")):
        if cibles and not any(c in src.name for c in cibles):
            continue
        construire(src, RACINE / "exercices" / "corriges" / (src.stem + "_corrige.ipynb"))
        construire(src, RACINE / "exercices" / "enonces" / (src.stem + ".ipynb"), retirer_solutions=True, executer_nb=False)


if __name__ == "__main__":
    main()

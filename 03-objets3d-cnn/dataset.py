"""Selection des images composites, identique aux TP LBP (01-02).

Copie volontaire de 01-objets3d-lbp/dataset.py : meme liste de categories,
meme fonction collect(), pour que le CNN soit evalue sur exactement le meme
split train/test (fixe au telechargement, voir 01-objets3d-lbp/manifest.csv).
"""

from pathlib import Path

CATEGORIES = ["shoe", "bottles_and_cans_and_cups", "bag", "board_games", "action_figures"]
LABELS = {nom: i for i, nom in enumerate(CATEGORIES)}


def list_images(directory):
    return sorted(Path(directory).glob("*.png"))


def collect(renders_root, split):
    """Renvoie [(chemin, label)] pour toutes les images composites d'un split."""
    selection = []
    for categorie in CATEGORIES:
        directory = Path(renders_root) / categorie / split
        for path in list_images(directory):
            selection.append((path, LABELS[categorie]))
    return selection

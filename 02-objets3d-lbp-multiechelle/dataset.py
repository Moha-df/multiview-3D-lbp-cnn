"""Selection des images composites et ecriture des fichiers de descripteurs.

Identique a 01-objets3d-lbp/dataset.py.
"""

from pathlib import Path

import numpy as np

CATEGORIES = ["shoe", "bottles_and_cans_and_cups", "bag", "board_games", "action_figures"]
LABELS = {nom: i for i, nom in enumerate(CATEGORIES)}


def list_images(directory):
    return sorted(Path(directory).glob("*.png"))


def collect(renders_root, split):
    """Renvoie [(chemin, label)] pour toutes les images composites d'un split.

    Le split (train/test) reprend celui fixe au telechargement des objets
    3D (voir 01-objets3d-lbp/manifest.csv) : aucun tirage aleatoire ici.
    """
    selection = []
    for categorie in CATEGORIES:
        directory = Path(renders_root) / categorie / split
        for path in list_images(directory):
            selection.append((path, LABELS[categorie]))
    return selection


def write_descriptors(path, descriptors, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        for descriptor, label in zip(descriptors, labels):
            f.write(" ".join(str(int(v)) for v in descriptor))
            f.write(" %d\n" % label)


def read_descriptors(path):
    data = np.loadtxt(path, dtype=np.int64, ndmin=2)
    return data[:, :-1], data[:, -1]

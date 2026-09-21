"""Descripteur LBP (Local Binary Pattern) 3x3, 256 motifs.

Identique a smart-parking-lbp-to-cnn/01-parking-lbp/lbp.py. Convention de parcours de la
fenetre 3x3 (spirale horaire depuis le coin superieur gauche), poids
binaires associes a chaque voisin :

      1    2    4
    128    c    8
     64   32   16

Le bit vaut 1 si le voisin est strictement superieur au pixel central.
"""

import cv2
import numpy as np

# (dy, dx, poids) dans l'ordre de la spirale horaire
NEIGHBOURS = (
    (-1, -1, 1),
    (-1, 0, 2),
    (-1, 1, 4),
    (0, 1, 8),
    (1, 1, 16),
    (1, 0, 32),
    (1, -1, 64),
    (0, -1, 128),
)


def lbp_codes(gray):
    """Carte des codes LBP d'une image en niveaux de gris (contours exclus)."""
    g = gray.astype(np.int16)
    h, w = g.shape
    if h < 3 or w < 3:
        raise ValueError("image trop petite pour une fenetre 3x3")

    centre = g[1:h - 1, 1:w - 1]
    codes = np.zeros(centre.shape, dtype=np.uint8)
    for dy, dx, poids in NEIGHBOURS:
        voisin = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        codes |= ((voisin > centre) * poids).astype(np.uint8)
    return codes


def lbp_histogram(gray):
    """Descripteur LBP : histogramme des 256 motifs (occurrences brutes)."""
    return np.bincount(lbp_codes(gray).ravel(), minlength=256).astype(np.int64)


def describe_file(path):
    """Charge une image, la convertit en niveaux de gris et renvoie son LBP."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("image illisible : %s" % path)
    return lbp_histogram(img)

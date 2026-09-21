"""Briques multi-echelle : decoupage en blocs, pyramide, voisinage circulaire.

Identique dans l'esprit a smart-parking-lbp-to-cnn/04-parking-lbp-multiechelle/multiscale.py.
Ces fonctions operent sur un tableau 2D quelconque (niveaux de gris, un plan
de couleur, ou l'image mosaique R|G|B) : la notion de "canal" est geree a part
dans color.py / descriptors.py.

Echelle spatiale
    L'image est decoupee en une grille de blocs. Un histogramme LBP H1, H2, ...
    est calcule sur chaque bloc, puis tous sont concatenes bout a bout : le
    descripteur conserve la position des motifs. Le mode pyramide combine les
    grilles 1x1, 2x2 et 4x4 dans un seul descripteur (gros grain + fin grain).

Echelle du voisinage
    Au lieu de la fenetre 3x3, les 8 voisins sont pris sur un cercle de rayon R
    (interpolation bilineaire). Le mode multirayon concatene R = 1, 2 et 3.
"""

import numpy as np

from lbp import lbp_codes

RAYONS = (1, 2, 3)
GRILLES_PYRAMIDE = (1, 2, 4)


def lbp_codes_circulaire(gray, rayon):
    """Carte des codes LBP a 8 voisins pris sur un cercle de rayon donne."""
    g = gray.astype(np.float64)
    hauteur, largeur = g.shape
    marge = int(np.ceil(rayon))
    if hauteur - 2 * marge < 1 or largeur - 2 * marge < 1:
        raise ValueError("image trop petite pour un rayon de %s" % rayon)

    lignes = np.arange(marge, hauteur - marge)[:, None]
    colonnes = np.arange(marge, largeur - marge)[None, :]
    centre = g[marge:hauteur - marge, marge:largeur - marge]
    codes = np.zeros(centre.shape, dtype=np.uint8)

    for bit in range(8):
        angle = 2.0 * np.pi * bit / 8.0
        y = lignes - rayon * np.sin(angle)
        x = colonnes + rayon * np.cos(angle)

        y0 = np.floor(y).astype(np.intp)
        x0 = np.floor(x).astype(np.intp)
        dy = y - y0
        dx = x - x0
        y1 = np.minimum(y0 + 1, hauteur - 1)
        x1 = np.minimum(x0 + 1, largeur - 1)

        voisin = (g[y0, x0] * (1 - dy) * (1 - dx) + g[y0, x1] * (1 - dy) * dx
                  + g[y1, x0] * dy * (1 - dx) + g[y1, x1] * dy * dx)
        codes |= ((voisin > centre).astype(np.uint8) << bit)

    return codes


def histogramme(codes):
    return np.bincount(codes.ravel(), minlength=256).astype(np.int64)


def blocs(codes, grille):
    """Decoupe la carte des codes en grille x grille blocs, ligne par ligne."""
    for bande in np.array_split(codes, grille, axis=0):
        for bloc in np.array_split(bande, grille, axis=1):
            yield bloc


def histogrammes_blocs(gray, grille):
    """Les grille^2 histogrammes H1, H2, ... dans l'ordre de lecture."""
    codes = lbp_codes(gray)
    return [histogramme(bloc) for bloc in blocs(codes, grille)]


def describe_grille(gray, grille):
    """Descripteur spatial : les histogrammes de blocs mis bout a bout."""
    return np.concatenate(histogrammes_blocs(gray, grille))


def describe_pyramide(gray):
    """Descripteur pyramidal : plusieurs grilles concatenees."""
    codes = lbp_codes(gray)
    morceaux = []
    for grille in GRILLES_PYRAMIDE:
        morceaux.extend(histogramme(bloc) for bloc in blocs(codes, grille))
    return np.concatenate(morceaux)


def describe_multirayon(gray):
    """Descripteur multi-rayon : un histogramme par rayon, concatenes."""
    return np.concatenate([histogramme(lbp_codes_circulaire(gray, rayon))
                           for rayon in RAYONS])


TAILLE_PYRAMIDE = sum(g * g for g in GRILLES_PYRAMIDE) * 256  # 21 * 256 = 5376

"""Descripteurs combinant l'echelle spatiale (parking-old/04) et la couleur
(parking-old/03), appliques aux images composites 6-vues.

Deux axes independants :

  canal   : gris (reference), mosaic (plans R|G|B juxtaposes, un seul LBP),
            perchannel (un LBP par plan, histogrammes concatenes)
  echelle : global (256 valeurs), grille2x2/3x3/4x4 (blocs), pyramide
            (1x1 + 2x2 + 4x4 combines), multirayon (voisinage circulaire
            R = 1, 2, 3)

MODES ne couvre pas toutes les combinaisons (ce serait surdimensionne pour
128 objets) mais les plus informatives : la reference grise a chaque echelle,
et la methode pyramidale - la plus efficace en 04 - declinee en couleur pour
verifier si la couleur lui apporte quelque chose ici.
"""

import numpy as np

from color import mosaic, planes_rgb, read_color, to_gray
from multiscale import (describe_grille, describe_multirayon, describe_pyramide,
                        histogramme)
from lbp import lbp_codes

TAILLE_UNITE = 256


def _global(gray):
    return histogramme(lbp_codes(gray))


def _pyramide_perchannel(bgr):
    r, g, b = planes_rgb(bgr)
    return np.concatenate([describe_pyramide(plan) for plan in (r, g, b)])


MODES = {
    "gris_global": lambda bgr: _global(to_gray(bgr)),
    "gris_grille2x2": lambda bgr: describe_grille(to_gray(bgr), 2),
    "gris_grille3x3": lambda bgr: describe_grille(to_gray(bgr), 3),
    "gris_grille4x4": lambda bgr: describe_grille(to_gray(bgr), 4),
    "gris_pyramide": lambda bgr: describe_pyramide(to_gray(bgr)),
    "gris_multirayon": lambda bgr: describe_multirayon(to_gray(bgr)),
    "couleur_mosaic_pyramide": lambda bgr: describe_pyramide(mosaic(bgr)),
    "couleur_perchannel_pyramide": _pyramide_perchannel,
}

TAILLE_DESCRIPTEUR = {
    "gris_global": 1 * TAILLE_UNITE,
    "gris_grille2x2": 4 * TAILLE_UNITE,
    "gris_grille3x3": 9 * TAILLE_UNITE,
    "gris_grille4x4": 16 * TAILLE_UNITE,
    "gris_pyramide": 21 * TAILLE_UNITE,
    "gris_multirayon": 3 * TAILLE_UNITE,
    "couleur_mosaic_pyramide": 21 * TAILLE_UNITE,
    "couleur_perchannel_pyramide": 3 * 21 * TAILLE_UNITE,
}

MODE_LABELS = {
    "gris_global": "gris, global (reference)",
    "gris_grille2x2": "gris, grille 2x2",
    "gris_grille3x3": "gris, grille 3x3",
    "gris_grille4x4": "gris, grille 4x4",
    "gris_pyramide": "gris, pyramide (1+2x2+4x4)",
    "gris_multirayon": "gris, multi-rayon (R=1,2,3)",
    "couleur_mosaic_pyramide": "couleur mosaique, pyramide",
    "couleur_perchannel_pyramide": "couleur par plan R/G/B, pyramide",
}


def describe_file(path, mode):
    return MODES[mode](read_color(path))

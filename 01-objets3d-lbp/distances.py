"""Metriques de comparaison d'histogrammes LBP.

Identique a smart-parking-lbp-to-cnn/02-parking-distances/distances.py. Chaque metrique
renvoie, pour un descripteur test et l'ensemble des descripteurs du
training, un vecteur de scores. `better` indique si le plus proche voisin
correspond au score minimal ("min") ou maximal ("max").
"""

import cv2
import numpy as np


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def l1(train, vecteur):
    """Somme des differences en valeurs absolues (methode de la seance 1)."""
    return np.abs(train - vecteur).sum(axis=1)


def l2_carre(train, vecteur):
    """Distance euclidienne au carre : somme des ecarts au carre."""
    ecart = train - vecteur
    return (ecart * ecart).sum(axis=1)


def l2(train, vecteur):
    """Distance euclidienne."""
    return np.sqrt(l2_carre(train, vecteur))


def _compare_hist(train, vecteur, methode):
    v = _f32(vecteur)
    return np.array([cv2.compareHist(_f32(ligne), v, methode) for ligne in train])


def bhattacharyya(train, vecteur):
    """cv2.HISTCMP_BHATTACHARYYA : 0 = identique, 1 = totalement different."""
    return _compare_hist(train, vecteur, cv2.HISTCMP_BHATTACHARYYA)


def chi2(train, vecteur):
    """cv2.HISTCMP_CHISQR."""
    return _compare_hist(train, vecteur, cv2.HISTCMP_CHISQR)


def correlation(train, vecteur):
    """cv2.HISTCMP_CORREL : similarite, 1 = identique."""
    return _compare_hist(train, vecteur, cv2.HISTCMP_CORREL)


def intersection(train, vecteur):
    """cv2.HISTCMP_INTERSECT : similarite, d'autant plus grande que proche."""
    return _compare_hist(train, vecteur, cv2.HISTCMP_INTERSECT)


# nom affiche -> (fonction, sens de la meilleure valeur)
METRICS = {
    "L1 (valeurs absolues)": (l1, "min"),
    "L2 (euclidienne)": (l2, "min"),
    "L2 au carre": (l2_carre, "min"),
    "Bhattacharyya (OpenCV)": (bhattacharyya, "min"),
    "Chi-2 (OpenCV)": (chi2, "min"),
    "Correlation (OpenCV)": (correlation, "max"),
    "Intersection (OpenCV)": (intersection, "max"),
}

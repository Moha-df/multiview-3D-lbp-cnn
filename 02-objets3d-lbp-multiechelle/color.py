"""Lecture couleur et strategies de mise en forme des 3 plans R, G, B.

Identique dans l'esprit a smart-parking-lbp-to-cnn/03-parking-lbp-couleur/color.py.
"""

import cv2


def read_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("image illisible : %s" % path)
    return img


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("image illisible : %s" % path)
    return img


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def planes_rgb(bgr):
    """Plans R, G, B de l'image (OpenCV charge les canaux dans l'ordre B, G, R)."""
    b, g, r = cv2.split(bgr)
    return r, g, b


def mosaic(bgr):
    """Image unique de largeur 3W : plans R, G puis B juxtaposes."""
    return cv2.hconcat(list(planes_rgb(bgr)))

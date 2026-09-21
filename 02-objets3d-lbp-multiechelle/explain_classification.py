"""Explication visuelle de la classification 1-NN pour un mode donne
(par defaut la methode pyramidale en niveaux de gris).

Pour chaque image test, on retrouve son plus proche voisin dans le training
et on construit une image cote a cote (image test au-dessus, voisin qui a
determine la prediction en-dessous), avec la categorie reelle, la categorie
predite et la distance. Le resultat est range dans :

    resultats/classification/<mode>/bien_classes/   -> predictions correctes
    resultats/classification/<mode>/mal_classes/     -> predictions fausses
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import CATEGORIES, collect
from descriptors import MODE_LABELS, MODES, describe_file

BANNER_H = 26
GAP = 6
MAX_NOM = 40  # les chemins Windows plafonnent a 260 caracteres


def _court(nom):
    return nom if len(nom) <= MAX_NOM else nom[:MAX_NOM]


def _font(size=15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _banner(width, text, bg):
    img = Image.new("RGB", (width, BANNER_H), bg)
    draw = ImageDraw.Draw(img)
    draw.text((6, 4), text, fill="white", font=_font())
    return img


def build_explanation(test_path, test_categorie, voisin_path, voisin_categorie, distance, correct, mode_label):
    test_img = Image.open(test_path).convert("RGB")
    voisin_img = Image.open(voisin_path).convert("RGB")
    largeur = max(test_img.width, voisin_img.width)

    verdict_bg = (40, 140, 40) if correct else (170, 40, 40)
    verdict_txt = "CORRECT" if correct else "ERREUR"

    bandeau_test = _banner(largeur, "TEST : %s   [mode : %s]" % (test_categorie, mode_label), (50, 50, 50))
    bandeau_voisin = _banner(
        largeur,
        "PLUS PROCHE VOISIN (train) : %s - distance = %.0f" % (voisin_categorie, distance),
        (50, 50, 50),
    )
    bandeau_verdict = _banner(largeur, "%s : prediction = %s" % (verdict_txt, voisin_categorie), verdict_bg)

    hauteur = (
        bandeau_test.height + test_img.height + GAP
        + bandeau_voisin.height + voisin_img.height + GAP
        + bandeau_verdict.height
    )
    composite = Image.new("RGB", (largeur, hauteur), (20, 20, 20))
    y = 0
    for bloc in (bandeau_test, test_img):
        composite.paste(bloc, (0, y))
        y += bloc.height
    y += GAP
    for bloc in (bandeau_voisin, voisin_img):
        composite.paste(bloc, (0, y))
        y += bloc.height
    y += GAP
    composite.paste(bandeau_verdict, (0, y))
    return composite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODES), default="gris_pyramide")
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--out-dir", default="resultats/classification")
    args = parser.parse_args()

    train = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")

    train_paths, train_labels = zip(*train)
    train_desc = np.stack([describe_file(p, args.mode) for p in train_paths])

    out_root = Path(args.out_dir) / args.mode
    for sous_dossier in ("bien_classes", "mal_classes"):
        dossier = out_root / sous_dossier
        if dossier.exists():
            shutil.rmtree(dossier)
        dossier.mkdir(parents=True)

    correct_total = 0
    for test_path, vraie_label in test:
        vecteur = describe_file(test_path, args.mode)
        distances = np.abs(train_desc - vecteur).sum(axis=1)
        idx = int(np.argmin(distances))
        predite_label = train_labels[idx]
        voisin_path = train_paths[idx]
        correct = predite_label == vraie_label
        correct_total += int(correct)

        composite = build_explanation(
            test_path, CATEGORIES[vraie_label],
            voisin_path, CATEGORIES[predite_label],
            distances[idx], correct, MODE_LABELS[args.mode],
        )
        sous_dossier = "bien_classes" if correct else "mal_classes"
        nom_fichier = "%s__%s__vs__%s.png" % (
            CATEGORIES[vraie_label], _court(Path(test_path).stem), _court(Path(voisin_path).stem)
        )
        composite.save(out_root / sous_dossier / nom_fichier)

    print("%d / %d images test correctement classees (mode %s)"
          % (correct_total, len(test), args.mode))
    print("Explications ecrites dans %s/bien_classes et %s/mal_classes" % (out_root, out_root))


if __name__ == "__main__":
    main()

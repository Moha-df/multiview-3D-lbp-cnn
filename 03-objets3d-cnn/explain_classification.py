"""Explication visuelle des predictions du CNN (tirage de demonstration).

Contrairement au LBP + plus-proche-voisin (01/02), le CNN n'a pas de "voisin"
explicite : a la place, on affiche les probabilites qu'il attribue a chaque
categorie (sortie softmax). Le resultat est range dans :

    resultats/classification/bien_classes/   -> predictions correctes
    resultats/classification/mal_classes/    -> predictions fausses
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cnn import load_images
from dataset import CATEGORIES, collect
from figstyle import SERIES, clean, plt

BANNER_H = 26
GAP = 6
MAX_NOM = 60


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


def _barres_probabilites(probas, largeur, hauteur=170):
    fig, ax = plt.subplots(figsize=(largeur / 100, hauteur / 100), dpi=100)
    y = np.arange(len(CATEGORIES))
    couleurs = [SERIES[2] if p == probas.max() else SERIES[0] for p in probas]
    ax.barh(y, probas * 100, color=couleurs)
    ax.set_yticks(y, CATEGORIES, fontsize=8)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("probabilite CNN (%)", fontsize=8)
    clean(ax, grid_axis="x")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf).convert("RGB")
    plt.close(fig)
    return img.resize((largeur, hauteur))


def build_explanation(test_img, test_categorie, probas, predite_categorie, correct):
    largeur = test_img.width

    verdict_bg = (40, 140, 40) if correct else (170, 40, 40)
    verdict_txt = "CORRECT" if correct else "ERREUR"

    bandeau_test = _banner(largeur, "TEST : %s" % test_categorie, (50, 50, 50))
    bandeau_proba = _banner(largeur, "PROBABILITES PREDITES PAR LE CNN", (50, 50, 50))
    barres = _barres_probabilites(probas, largeur)
    bandeau_verdict = _banner(
        largeur, "%s : prediction = %s (%.0f %%)"
        % (verdict_txt, predite_categorie, 100 * probas.max()), verdict_bg,
    )

    hauteur = bandeau_test.height + test_img.height + GAP + bandeau_proba.height + barres.height + GAP + bandeau_verdict.height
    composite = Image.new("RGB", (largeur, hauteur), (20, 20, 20))
    y = 0
    for bloc in (bandeau_test, test_img):
        composite.paste(bloc, (0, y))
        y += bloc.height
    y += GAP
    for bloc in (bandeau_proba, barres):
        composite.paste(bloc, (0, y))
        y += bloc.height
    y += GAP
    composite.paste(bandeau_verdict, (0, y))
    return composite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="resultats/classification")
    args = parser.parse_args()

    training = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from cnn import build_model, stratified_split

    tf.keras.backend.clear_session()
    x_train_full, y_train_full = load_images(training)
    x_test, y_test = load_images(test)
    (x_train, y_train), (x_val, y_val) = stratified_split(x_train_full, y_train_full, 0.2, args.seed)

    modele = build_model(augmentation=True, seed=args.seed)
    arret = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    modele.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=40,
              batch_size=16, callbacks=[arret], verbose=0)
    probas_test = modele.predict(x_test, batch_size=32, verbose=0)

    out_root = Path(args.out_dir)
    for sous_dossier in ("bien_classes", "mal_classes"):
        dossier = out_root / sous_dossier
        if dossier.exists():
            shutil.rmtree(dossier)
        dossier.mkdir(parents=True)

    correct_total = 0
    for (path, vraie_label), probas in zip(test, probas_test):
        predite_label = int(probas.argmax())
        correct = predite_label == vraie_label
        correct_total += int(correct)

        composite = build_explanation(
            Image.open(path).convert("RGB"), CATEGORIES[vraie_label],
            probas, CATEGORIES[predite_label], correct,
        )
        sous_dossier = "bien_classes" if correct else "mal_classes"
        nom_fichier = "%s__%s.png" % (CATEGORIES[vraie_label], Path(path).stem[:MAX_NOM])
        composite.save(out_root / sous_dossier / nom_fichier)

    print("%d / %d images test correctement classees (CNN, graine %d)"
          % (correct_total, len(test), args.seed))
    print("Explications ecrites dans %s/bien_classes et %s/mal_classes" % (out_root, out_root))


if __name__ == "__main__":
    main()

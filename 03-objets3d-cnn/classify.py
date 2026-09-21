"""Entraine et evalue le CNN sur un seul tirage (mise en oeuvre en ligne de commande)."""

import argparse

import numpy as np

from cnn import IMG_H, IMG_W, train_and_evaluate
from dataset import CATEGORIES, collect


def confusion(y_test, predictions):
    print("Matrice de confusion (lignes = verite, colonnes = prediction)")
    entete = "%-28s" % "" + "".join("%14s" % nom[:14] for nom in CATEGORIES)
    print(entete)
    for i, nom in enumerate(CATEGORIES):
        ligne = [int(((y_test == i) & (predictions == j)).sum()) for j in range(len(CATEGORIES))]
        print("%-28s" % nom + "".join("%14d" % v for v in ligne))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-w", type=int, default=IMG_W)
    parser.add_argument("--img-h", type=int, default=IMG_H)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    training = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")

    taux, historique, _, y_test, predictions = train_and_evaluate(
        training, test, seed=args.seed, img_w=args.img_w, img_h=args.img_h,
        epochs=args.epochs, patience=args.patience, verbose=2)

    print("\nImages test : %d" % len(test))
    print("Epoques effectuees     : %d" % len(historique["loss"]))
    print("Taux de reconnaissance : %.2f %%" % taux)
    print()
    confusion(np.asarray(y_test), np.asarray(predictions))


if __name__ == "__main__":
    main()

"""Classification 1 plus proche voisin (distance L1) des descripteurs."""

import argparse

import numpy as np

from dataset import CATEGORIES, read_descriptors
from descriptors import MODES


def classify(train, train_labels, test):
    predictions = np.empty(len(test), dtype=np.int64)
    for i, vecteur in enumerate(test):
        distances = np.abs(train - vecteur).sum(axis=1)
        predictions[i] = train_labels[int(np.argmin(distances))]
    return predictions


def evaluate(training_path, test_path):
    train, train_labels = read_descriptors(training_path)
    test, test_labels = read_descriptors(test_path)
    predictions = classify(train, train_labels, test)
    return int((predictions == test_labels).sum()), len(test_labels), train.shape[1]


def confusion(training_path, test_path):
    train, train_labels = read_descriptors(training_path)
    test, test_labels = read_descriptors(test_path)
    predictions = classify(train, train_labels, test)

    print("Images test correctement reconnues : %d / %d"
          % ((predictions == test_labels).sum(), len(test_labels)))
    print("Taux de reconnaissance             : %.2f %%"
          % (100.0 * (predictions == test_labels).mean()))
    print()
    print("Matrice de confusion (lignes = verite, colonnes = prediction)")
    entete = "%-28s" % "" + "".join("%14s" % nom[:14] for nom in CATEGORIES)
    print(entete)
    for i, nom in enumerate(CATEGORIES):
        ligne = [int(((test_labels == i) & (predictions == j)).sum()) for j in range(len(CATEGORIES))]
        print("%-28s" % nom + "".join("%14d" % v for v in ligne))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODES), default="gris_pyramide")
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    base = "%s/%s" % (args.out_dir, args.mode)
    confusion("%s/training.txt" % base, "%s/test.txt" % base)


if __name__ == "__main__":
    main()

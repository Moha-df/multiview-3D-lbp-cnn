"""Classification 1 plus proche voisin des descripteurs LBP (5 categories).

Meme protocole que smart-parking-lbp-to-cnn/01-parking-lbp : distance L1 (somme des
differences en valeurs absolues) entre le descripteur test et chacun des
descripteurs training, label du plus proche voisin retenu.
"""

import argparse

import numpy as np

from dataset import CATEGORIES, read_descriptors


def classify(train, train_labels, test):
    """Label predit pour chaque ligne de test (distance L1, 1-NN)."""
    predictions = np.empty(len(test), dtype=np.int64)
    for i, vecteur in enumerate(test):
        distances = np.abs(train - vecteur).sum(axis=1)
        predictions[i] = train_labels[int(np.argmin(distances))]
    return predictions


def report(predictions, labels):
    correct = int((predictions == labels).sum())
    total = len(labels)
    print("Images test correctement reconnues : %d / %d" % (correct, total))
    print("Taux de reconnaissance             : %.2f %%" % (100.0 * correct / total))
    print()
    print("Matrice de confusion (lignes = verite, colonnes = prediction)")
    entete = "%-28s" % "" + "".join("%14s" % nom[:14] for nom in CATEGORIES)
    print(entete)
    for i, nom in enumerate(CATEGORIES):
        ligne = [int(((labels == i) & (predictions == j)).sum()) for j in range(len(CATEGORIES))]
        print("%-28s" % nom + "".join("%14d" % v for v in ligne))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="resultats/training.txt")
    parser.add_argument("--test", default="resultats/test.txt")
    args = parser.parse_args()

    train, train_labels = read_descriptors(args.training)
    test, test_labels = read_descriptors(args.test)
    report(classify(train, train_labels, test), test_labels)


if __name__ == "__main__":
    main()

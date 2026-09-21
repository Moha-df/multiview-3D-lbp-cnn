"""Comparaison des metriques de distance sur la classification LBP des objets.

Le protocole est identique pour toutes les metriques : pour chaque descripteur
du jeu de test, on cherche son plus proche voisin parmi les descripteurs du
training et on lui attribue le label de ce voisin. Seule la metrique change.
"""

import argparse
import time

import numpy as np

from dataset import read_descriptors
from distances import METRICS


def classify(train, train_labels, test, fonction, sens):
    choisir = np.argmin if sens == "min" else np.argmax
    predictions = np.empty(len(test), dtype=np.int64)
    for i, vecteur in enumerate(test):
        predictions[i] = train_labels[int(choisir(fonction(train, vecteur)))]
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="resultats/training.txt")
    parser.add_argument("--test", default="resultats/test.txt")
    args = parser.parse_args()

    train, train_labels = read_descriptors(args.training)
    test, test_labels = read_descriptors(args.test)
    print("training : %d descripteurs, test : %d descripteurs\n"
          % (len(train), len(test)))

    entete = "%-24s %10s %12s %10s" % ("Metrique", "Correct", "Taux", "Duree")
    print(entete)
    print("-" * len(entete))

    for nom, (fonction, sens) in METRICS.items():
        debut = time.perf_counter()
        predictions = classify(train, train_labels, test, fonction, sens)
        duree = time.perf_counter() - debut
        correct = int((predictions == test_labels).sum())
        print("%-24s %5d/%-4d %11.2f %% %9.2f s"
              % (nom, correct, len(test_labels), 100.0 * correct / len(test_labels), duree))


if __name__ == "__main__":
    main()

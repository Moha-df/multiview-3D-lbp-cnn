"""Compare toutes les strategies (echelle x couleur) sur le meme split.

Ecrit resultats/comparaison.csv (repris par figures.py) et affiche un tableau
recapitulatif.
"""

import argparse
import csv
from pathlib import Path

from build_dataset import build
from classify import evaluate
from dataset import collect
from descriptors import MODE_LABELS, MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    training = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")

    resultats = []
    for mode in MODES:
        base = "%s/%s" % (args.out_dir, mode)
        build(training, mode, "%s/training.txt" % base)
        build(test, mode, "%s/test.txt" % base)
        resultats.append((mode,) + evaluate("%s/training.txt" % base, "%s/test.txt" % base))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open("%s/comparaison.csv" % args.out_dir, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(("mode", "libelle", "descripteur", "correct", "total", "taux"))
        for mode, correct, total, taille in resultats:
            writer.writerow((mode, MODE_LABELS[mode], taille, correct, total,
                             "%.2f" % (100.0 * correct / total)))

    entete = "\n%-30s %13s %10s %10s" % ("Mode", "Descripteur", "Correct", "Taux")
    print(entete)
    print("-" * (len(entete) - 1))
    for mode, correct, total, taille in sorted(resultats, key=lambda r: -r[1] / r[2]):
        print("%-30s %9d val %5d/%-4d %8.2f %%"
              % (MODE_LABELS[mode], taille, correct, total, 100.0 * correct / total))


if __name__ == "__main__":
    main()

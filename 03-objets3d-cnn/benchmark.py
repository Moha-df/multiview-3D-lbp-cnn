"""Evalue le CNN sur plusieurs graines d'entrainement independantes.

A la difference de parking-old/05-parking-cnn (qui repioche 200+200
imagettes dans un grand pool a chaque tirage), notre jeu train/test est fixe
(94 / 34 images, voir 01-objets3d-lbp/manifest.csv) : il n'y a pas assez
d'objets par categorie pour retirer des splits differents sans les
recouvrir. Les "tirages" ici varient uniquement l'initialisation des poids,
l'augmentation de donnees et le decoupage train/validation interne -- ce qui
quantifie la variance propre a l'entrainement, pas celle de l'echantillonnage
des donnees.

Les taux sont enregistres dans resultats/benchmark.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from cnn import train_and_evaluate
from dataset import collect


def run(renders_root, n_runs, sortie, **kwargs):
    training = collect(renders_root, "train")
    test = collect(renders_root, "test")

    taux_liste = []
    for run_id in range(n_runs):
        taux, historique, _, _, _ = train_and_evaluate(training, test, seed=run_id, **kwargs)
        taux_liste.append((taux, len(historique["loss"])))
        print("run %d/%d : %.2f %% (%d epoques)"
              % (run_id + 1, n_runs, taux, len(historique["loss"])), flush=True)

    Path(sortie).parent.mkdir(parents=True, exist_ok=True)
    with open(sortie, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(("run", "taux", "epoques"))
        for run_id, (taux, epoques) in enumerate(taux_liste):
            writer.writerow((run_id, "%.2f" % taux, epoques))
    return [t for t, _ in taux_liste]


def report(taux_liste):
    v = np.array(taux_liste)
    print("\nTaux de reconnaissance sur %d graines d'entrainement\n" % len(v))
    print("Moyenne   : %.2f %%" % v.mean())
    print("Ecart-type: %.2f" % v.std())
    print("Min / Max : %.2f %% / %.2f %%" % (v.min(), v.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    taux_liste = run(args.renders_root, args.runs, "%s/benchmark.csv" % args.out_dir,
                     epochs=args.epochs, patience=args.patience)
    report(taux_liste)


if __name__ == "__main__":
    main()

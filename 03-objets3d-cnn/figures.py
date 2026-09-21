"""Genere les figures du dossier resultats/."""

import argparse
import csv

import numpy as np

from cnn import train_and_evaluate
from dataset import CATEGORIES, collect
from figstyle import INK_SECOND, SERIES, clean, plt

# Reference : taux obtenus dans les etudes LBP (voir leurs README/resultats)
LBP_GLOBAL = 76.47
LBP_PYRAMIDE = 91.18


def lire_benchmark(chemin):
    lignes = []
    with open(chemin, encoding="ascii") as f:
        for ligne in csv.DictReader(f):
            lignes.append(ligne)
    return lignes


def figure_courbes(historique, sortie):
    """Courbes de perte et de precision, entrainement vs validation."""
    epoques = range(1, len(historique["loss"]) + 1)

    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    gauche.plot(epoques, historique["loss"], color=SERIES[0], label="training")
    gauche.plot(epoques, historique["val_loss"], color=SERIES[1], label="validation")
    gauche.set_xlabel("epoque")
    gauche.set_ylabel("perte (loss)")
    gauche.set_title("Perte")
    gauche.legend()
    clean(gauche)

    droite.plot(epoques, [100 * v for v in historique["accuracy"]], color=SERIES[0], label="training")
    droite.plot(epoques, [100 * v for v in historique["val_accuracy"]], color=SERIES[1], label="validation")
    droite.set_xlabel("epoque")
    droite.set_ylabel("precision (%)")
    droite.set_title("Precision")
    droite.legend()
    clean(droite)

    fig.suptitle("Entrainement du CNN (tirage de demonstration, arret anticipe)",
                 x=0.02, y=1.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def figure_confusion(y_test, predictions, sortie):
    n = len(CATEGORIES)
    matrice = np.zeros((n, n), dtype=int)
    for verite, prediction in zip(y_test, predictions):
        matrice[verite, prediction] += 1

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(matrice, cmap="Blues", vmin=0)
    ax.set_xticks(range(n), CATEGORIES, rotation=35, ha="right")
    ax.set_yticks(range(n), CATEGORIES)
    ax.set_xlabel("categorie predite")
    ax.set_ylabel("categorie reelle")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            valeur = matrice[i, j]
            if valeur == 0:
                continue
            couleur = "white" if valeur > matrice.max() / 2 else INK_SECOND
            ax.text(j, i, str(valeur), ha="center", va="center", color=couleur, fontweight="bold")
    taux = 100.0 * np.trace(matrice) / matrice.sum()
    ax.set_title("Matrice de confusion - CNN, tirage de demonstration (%.2f %%)" % taux, pad=14)
    fig.tight_layout()
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def figure_benchmark(chemin_csv, sortie):
    """Dot plot des 10 graines d'entrainement, avec reperes LBP."""
    lignes = lire_benchmark(chemin_csv)
    taux = np.array([float(l["taux"]) for l in lignes])

    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.axvline(LBP_GLOBAL, color=SERIES[1], linewidth=1.5, linestyle=(0, (5, 3)), zorder=1)
    ax.annotate("LBP global : %.2f %%" % LBP_GLOBAL, (LBP_GLOBAL, 0.85),
                xycoords=("data", "axes fraction"), color=SERIES[1], fontsize=8,
                fontweight="bold", ha="center")
    ax.axvline(LBP_PYRAMIDE, color=SERIES[2], linewidth=1.5, linestyle=(0, (5, 3)), zorder=1)
    ax.annotate("LBP pyramide : %.2f %%" % LBP_PYRAMIDE, (LBP_PYRAMIDE, 0.85),
                xycoords=("data", "axes fraction"), color=SERIES[2], fontsize=8,
                fontweight="bold", ha="center")

    ax.scatter(taux, np.zeros_like(taux), s=60, color=SERIES[0], alpha=0.6,
              linewidths=0, zorder=3)
    ax.scatter(taux.mean(), 0, s=140, color=SERIES[0], edgecolors="#fcfcfb",
              linewidths=2, zorder=4)
    ax.annotate("moyenne %.2f %%" % taux.mean(), (taux.mean(), 0),
               textcoords="offset points", xytext=(0, 14), ha="center",
               color=SERIES[0], fontsize=9, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("taux de reconnaissance sur le jeu test (%)")
    ax.set_xlim(50, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("CNN sur %d graines d'entrainement : moyenne %.2f %%, "
                 "ecart-type %.2f (min %.2f, max %.2f)"
                 % (len(taux), taux.mean(), taux.std(), taux.min(), taux.max()), pad=22)
    clean(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    training = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")
    _, historique, _, y_test, predictions = train_and_evaluate(
        training, test, seed=args.seed, verbose=0)

    figure_courbes(historique, "%s/courbes-entrainement.png" % args.out_dir)
    figure_confusion(np.asarray(y_test), np.asarray(predictions),
                     "%s/confusion-cnn.png" % args.out_dir)
    figure_benchmark("%s/benchmark.csv" % args.out_dir, "%s/benchmark-cnn.png" % args.out_dir)


if __name__ == "__main__":
    main()

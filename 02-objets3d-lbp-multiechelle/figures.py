"""Genere les figures du dossier resultats/."""

import argparse
import csv

import numpy as np

from classify import classify
from color import read_gray
from dataset import CATEGORIES, collect, read_descriptors
from descriptors import MODE_LABELS
from figstyle import AXIS, INK_SECOND, SERIES, clean, plt
from multiscale import RAYONS, histogramme, histogrammes_blocs, lbp_codes_circulaire


def lire_comparaison(chemin):
    lignes = []
    with open(chemin, encoding="ascii") as f:
        for ligne in csv.DictReader(f):
            lignes.append(ligne)
    return lignes


def figure_decoupage(chemin_image, grille, sortie):
    """L'image decoupee en blocs, un histogramme par bloc, puis la concatenation."""
    gris = read_gray(chemin_image)
    histos = histogrammes_blocs(gris, grille)
    plafond = max(h[1:].max() for h in histos)

    fig = plt.figure(figsize=(9.8, 6.4))
    structure = fig.add_gridspec(2, 2, height_ratios=(2.1, 1),
                                 width_ratios=(1, 1.5), hspace=0.42, wspace=0.18)

    ax = fig.add_subplot(structure[0, 0])
    ax.imshow(gris, cmap="gray", vmin=0, vmax=255)
    hauteur, largeur = gris.shape
    for k in range(1, grille):
        ax.axhline(k * hauteur / grille - 0.5, color=SERIES[1], linewidth=1.5)
        ax.axvline(k * largeur / grille - 0.5, color=SERIES[1], linewidth=1.5)
    for i in range(grille):
        for j in range(grille):
            ax.text((j + 0.5) * largeur / grille, (i + 0.5) * hauteur / grille,
                    "H%d" % (i * grille + j + 1), ha="center", va="center",
                    color=SERIES[1], fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, linewidth=0))
    ax.set_title("1. L'image (6 vues) est decoupee en %d x %d blocs" % (grille, grille))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    cellules = structure[0, 1].subgridspec(grille, grille, hspace=0.35, wspace=0.12)
    for k, histo in enumerate(histos):
        sous = fig.add_subplot(cellules[k // grille, k % grille])
        sous.fill_between(np.arange(256), histo, color=SERIES[0], linewidth=0)
        sous.set_xlim(0, 255)
        sous.set_ylim(0, plafond)
        sous.set_xticks([])
        sous.set_yticks([])
        sous.grid(False)
        for cote in ("top", "right"):
            sous.spines[cote].set_visible(False)
        sous.set_title("H%d" % (k + 1), fontsize=7, pad=2)

    ax = fig.add_subplot(structure[1, :])
    concatene = np.concatenate(histos)
    ax.fill_between(np.arange(len(concatene)), concatene, color=SERIES[0], linewidth=0)
    for k in range(1, len(histos)):
        ax.axvline(k * 256, color=AXIS, linewidth=1)
    ax.set_xlim(0, len(concatene))
    ax.set_ylim(0, plafond)
    ax.set_xlabel("position dans le descripteur")
    ax.set_ylabel("occurrences")
    ax.set_title("2. Les %d histogrammes mis bout a bout : le descripteur, %d valeurs"
                 % (len(histos), len(concatene)))
    clean(ax)

    fig.suptitle("Methode pyramidale (echelle spatiale) : decouper, decrire, concatener",
                 x=0.02, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.935, "Le descripteur global perd la position des motifs ; "
             "en decrivant chaque bloc separement, elle est conservee.",
             ha="left", fontsize=8.5, color=INK_SECOND)
    fig.subplots_adjust(top=0.87, bottom=0.09, left=0.07, right=0.97)
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def figure_rayons(chemin_image, sortie):
    """Effet du rayon du voisinage sur la carte des codes et l'histogramme."""
    gris = read_gray(chemin_image)

    fig, axes = plt.subplots(2, len(RAYONS), figsize=(9.0, 4.6),
                             gridspec_kw={"height_ratios": (1.5, 1)})
    cartes = [lbp_codes_circulaire(gris, rayon) for rayon in RAYONS]
    plafond = max(histogramme(c)[1:].max() for c in cartes)

    for colonne, (rayon, carte) in enumerate(zip(RAYONS, cartes)):
        haut = axes[0, colonne]
        haut.imshow(carte, cmap="gray", vmin=0, vmax=255)
        haut.set_title("rayon R = %d" % rayon)
        haut.set_xticks([])
        haut.set_yticks([])
        haut.grid(False)

        bas = axes[1, colonne]
        bas.fill_between(np.arange(256), histogramme(carte), color=SERIES[0], linewidth=0)
        bas.set_xlim(0, 255)
        bas.set_ylim(0, plafond)
        bas.set_xlabel("code du motif")
        if colonne == 0:
            bas.set_ylabel("occurrences")
        bas.margins(y=0.18)
    clean(bas)

    fig.suptitle("Echelle du voisinage : les 8 voisins sur un cercle de rayon R",
                 x=0.02, y=0.99, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.11, left=0.07, right=0.97, hspace=0.35)
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def figure_comparaison_modes(chemin_csv, sortie):
    """Dot plot du taux de reconnaissance de chaque mode, un seul essai chacun."""
    lignes = lire_comparaison(chemin_csv)
    lignes.sort(key=lambda l: float(l["taux"]))

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for y, ligne in enumerate(lignes):
        couleur = SERIES[1] if ligne["mode"] == "gris_global" else SERIES[0]
        if "pyramide" in ligne["mode"]:
            couleur = SERIES[2]
        taux = float(ligne["taux"])
        ax.plot((0, taux), (y, y), color=AXIS, linewidth=1, zorder=1)
        ax.scatter(taux, y, s=100, color=couleur, edgecolors="#fcfcfb",
                   linewidths=2, zorder=4)
        ax.annotate("%.2f %% (%d/%d)" % (taux, int(ligne["correct"]), int(ligne["total"])),
                    (taux, y), textcoords="offset points", xytext=(10, 0),
                    va="center", color=couleur, fontsize=8.5, fontweight="bold")

    ax.set_yticks(range(len(lignes)),
                  ["%s\n%s val" % (MODE_LABELS[l["mode"]], l["descripteur"]) for l in lignes])
    ax.set_xlabel("taux de reconnaissance sur le jeu test (%)")
    ax.set_xlim(0, 105)
    ax.set_title("La methode pyramidale (aqua) egale ou depasse toutes les "
                 "autres variantes, couleur comprise", pad=22)
    ax.annotate("orange = descripteur global de reference  |  aqua = pyramide (gris et couleur)",
                (0, 1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 6), color=INK_SECOND, fontsize=8)
    ax.set_ylim(-0.7, len(lignes) - 0.3)
    clean(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def figure_confusion(mode, out_dir, sortie):
    """Matrice de confusion (heatmap) du mode donne."""
    base = "%s/%s" % (out_dir, mode)
    train, train_labels = read_descriptors("%s/training.txt" % base)
    test, test_labels = read_descriptors("%s/test.txt" % base)
    predictions = classify(train, train_labels, test)

    n = len(CATEGORIES)
    matrice = np.zeros((n, n), dtype=int)
    for verite, prediction in zip(test_labels, predictions):
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
            ax.text(j, i, str(valeur), ha="center", va="center",
                    color=couleur, fontweight="bold")
    taux = 100.0 * np.trace(matrice) / matrice.sum()
    ax.set_title("Matrice de confusion - %s (%.2f %%)" % (MODE_LABELS[mode], taux), pad=14)
    fig.tight_layout()
    fig.savefig(sortie)
    plt.close(fig)
    print("figure :", sortie)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--mode-confusion", default="gris_pyramide")
    parser.add_argument("--grille", type=int, default=4)
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    exemple, _ = collect(args.renders_root, "train")[0]

    figure_decoupage(exemple, args.grille, "%s/decoupage-blocs.png" % args.out_dir)
    figure_rayons(exemple, "%s/rayons.png" % args.out_dir)
    figure_comparaison_modes("%s/comparaison.csv" % args.out_dir,
                             "%s/comparaison-modes.png" % args.out_dir)
    figure_confusion(args.mode_confusion, args.out_dir,
                     "%s/confusion-%s.png" % (args.out_dir, args.mode_confusion))


if __name__ == "__main__":
    main()

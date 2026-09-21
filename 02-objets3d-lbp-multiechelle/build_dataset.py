"""Genere les descripteurs (echelle x couleur) du jeu training et du jeu test.

Les fichiers sont ecrits dans resultats/<mode>/training.txt et
resultats/<mode>/test.txt : une ligne par image, les valeurs du descripteur
puis le label de categorie (0 a 4, voir dataset.CATEGORIES). Le split
train/test reprend celui fixe au telechargement des objets (voir
01-objets3d-lbp/manifest.csv) : aucun tirage aleatoire ici.
"""

import argparse

from dataset import CATEGORIES, collect, write_descriptors
from descriptors import MODES, describe_file


def build(selection, mode, output):
    descriptors = [describe_file(path, mode) for path, _ in selection]
    labels = [label for _, label in selection]
    write_descriptors(output, descriptors, labels)
    print("%s : %d lignes de %d valeurs" % (output, len(labels), len(descriptors[0])))
    for i, nom in enumerate(CATEGORIES):
        print("    %-28s %d" % (nom, labels.count(i)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODES), default="gris_pyramide")
    parser.add_argument("--renders-root", default="../01-objets3d-lbp/data/renders")
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    training = collect(args.renders_root, "train")
    test = collect(args.renders_root, "test")

    base = "%s/%s" % (args.out_dir, args.mode)
    build(training, args.mode, "%s/training.txt" % base)
    build(test, args.mode, "%s/test.txt" % base)


if __name__ == "__main__":
    main()

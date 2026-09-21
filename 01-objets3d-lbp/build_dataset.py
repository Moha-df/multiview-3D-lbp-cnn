"""Genere resultats/training.txt et resultats/test.txt a partir des images
composites 6-vues rendues par render_views.py.

Chaque ligne contient les 256 valeurs du descripteur LBP de l'image composite,
suivies du label de categorie (0 a 4, voir dataset.CATEGORIES). Le split
train/test reprend celui fixe au telechargement (voir manifest.csv) : aucun
tirage aleatoire ici.
"""

import argparse

from dataset import CATEGORIES, collect, write_descriptors
from lbp import describe_file


def build(selection, output):
    descriptors = [describe_file(path) for path, _ in selection]
    labels = [label for _, label in selection]
    write_descriptors(output, descriptors, labels)
    print("%s : %d lignes" % (output, len(labels)))
    for i, nom in enumerate(CATEGORIES):
        print("    %-28s %d" % (nom, labels.count(i)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", default="data/renders")
    parser.add_argument("--out-dir", default="resultats")
    args = parser.parse_args()

    build(collect(args.renders_root, "train"), "%s/training.txt" % args.out_dir)
    build(collect(args.renders_root, "test"), "%s/test.txt" % args.out_dir)


if __name__ == "__main__":
    main()

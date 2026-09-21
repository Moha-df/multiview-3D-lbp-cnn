# LBP multi-echelle (pyramidal) et couleur

Reprend `01-objets3d-lbp` en combinant deux ameliorations testees separement
dans `parking-old` :

- **echelle spatiale** (`parking-old/04-...`) : au lieu d'un histogramme LBP
  global (qui perd toute position), l'image est decoupee en blocs, un
  histogramme par bloc, tous concatenes. Le mode **pyramide** combine les
  grilles 1x1 + 2x2 + 4x4 dans un seul descripteur.
- **couleur** (`parking-old/03-...`) : LBP calcule soit sur la mosaique des
  plans R, G, B juxtaposes, soit un LBP par plan avec histogrammes
  concatenes.

Les images utilisees sont celles deja rendues par `01-objets3d-lbp/render_views.py`
(`data/renders/`, non versionnees) ; le split train/test est celui fixe au
telechargement (`01-objets3d-lbp/manifest.csv`).

## Pipeline

```
python compare_modes.py    # construit tous les modes, evalue, ecrit resultats/comparaison.csv
python classify.py --mode gris_pyramide       # matrice de confusion pour un mode
python figures.py                              # les graphes (voir plus bas)
python explain_classification.py --mode gris_pyramide  # explications visuelles
```

## Resultats

Un seul essai par mode (le train/test est fixe, pas de tirages repetes comme
dans `parking-old` qui piochait dans un grand pool) :

| Mode | Descripteur | Taux |
| --- | --- | --- |
| gris, global (reference) | 256 val | 76,47 % (26/34) |
| gris, multi-rayon (R=1,2,3) | 768 val | 85,29 % (29/34) |
| gris, grille 2x2 | 1024 val | 91,18 % (31/34) |
| gris, grille 3x3 | 2304 val | 91,18 % (31/34) |
| gris, grille 4x4 | 4096 val | 91,18 % (31/34) |
| **gris, pyramide (1+2x2+4x4)** | 5376 val | **91,18 % (31/34)** |
| couleur mosaique, pyramide | 5376 val | 91,18 % (31/34) |
| couleur par plan R/G/B, pyramide | 16128 val | 91,18 % (31/34) |

Meme constat que dans `parking-old/04` : d'ecouper l'image en blocs
(l'echelle spatiale) apporte un vrai gain (76,47 % -> 91,18 %), des la
grille 2x2. Et meme constat que dans `parking-old/03` : **la couleur
n'apporte rien de plus** une fois l'echelle spatiale en place - le LBP est
invariant aux changements monotones d'intensite, et les trois plans R/G/B
de nos rendus portent la meme texture. Le multi-rayon (voisinage circulaire)
aide un peu par rapport au global mais nettement moins que le decoupage en
blocs.

## Graphes (`resultats/`)

- `decoupage-blocs.png` - illustre la methode pyramidale : l'image decoupee
  en blocs, un histogramme par bloc, la concatenation finale.
- `rayons.png` - effet du rayon du voisinage circulaire sur la carte des
  codes et l'histogramme.
- `comparaison-modes.png` - taux de reconnaissance de chaque mode (dot plot).
- `confusion-gris_pyramide.png` - matrice de confusion (heatmap) du mode
  retenu.

## Explications visuelles (`resultats/classification/<mode>/`)

Comme dans `01-objets3d-lbp`, pour chaque image test on affiche l'image et
son plus proche voisin d'entrainement (celui qui a determine la
prediction), rangees dans `bien_classes/` ou `mal_classes/`. Avec le mode
pyramidal, il ne reste que 3 erreurs sur 34 (contre 8 avec le LBP global) ;
l'une d'elles persiste malgre le decoupage en blocs : une botte UGG
confondue avec une figurine Ninja Turtle, dont la silhouette repliee produit
localement des motifs LBP proches de ceux de la botte.

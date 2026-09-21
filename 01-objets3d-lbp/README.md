# Objets 3D categorises (Google Scanned Objects)

Sous-ensemble du dataset [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)
(objets reels scannes, maillage `.obj` + texture reelle, licence CC-BY 4.0),
telecharge via l'API REST de [Gazebo Fuel](https://fuel.gazebosim.org/).

## Composition

| Categorie | Train | Test | Total disponible |
| --- | --- | --- | --- |
| Shoe | 25 | 8 | 254 |
| Bottles and Cans and Cups | 25 | 8 | 53 |
| Bag | 20 | 8 | 28 |
| Board Games | 12 | 5 | 17 |
| Action Figures | 12 | 5 | 17 |

Soit 128 objets (94 train / 34 test), ~1,2 Go. Le detail exact (nom d'objet,
categorie, split, taille, URL source) est dans [`manifest.csv`](manifest.csv).

Le split train/test est tire aleatoirement mais de maniere reproductible
(graine fixe) par `download_dataset.js` : deux executions produisent le meme
manifeste.

## Structure

```
data/raw/<categorie>/<train|test>/<nom_objet>.zip
```

Chaque zip contient `meshes/model.obj`, `meshes/model.mtl` et
`materials/textures/texture.png`. Le dossier `data/` n'est pas versionne
(voir `.gitignore` a la racine du depot) : il se regenere avec

```
node download_dataset.js
```

## Pipeline

```
pip install -r requirements.txt

node download_dataset.js       # deja fait : remplit data/raw/
python render_views.py         # rend 6 vues/objet, concatenees -> data/renders/
python build_dataset.py        # descripteurs LBP -> resultats/{training,test}.txt
python classify.py             # 1-plus-proche-voisin, distance L1
python compare.py              # meme protocole, 7 metriques de distance
python explain_classification.py  # explications visuelles (voir plus bas)
```

### Rendu 6 vues

`render_views.py` charge chaque `model.obj` (maillage + texture reelle),
le centre, le mets a l'echelle, puis le "photographie" selon les 6
directions cardinales (face, dos, droite, gauche, dessus, dessous) avec
`pyrender` (rendu offscreen, camera orthographique, lumiere directionnelle
alignee sur chaque vue). Les 6 vues 128x128 sont concatenees horizontalement
en une image composite 768x128, stockee dans `data/renders/<categorie>/<train|test>/`
(non versionne, regenerable).

### Classification LBP (1-plus-proche-voisin)

Meme descripteur et meme protocole que [`01-parking-lbp`](https://github.com/Moha-df/smart-parking-lbp-to-cnn/tree/main/01-parking-lbp) :
histogramme LBP 256 motifs (fenetre 3x3) sur l'image composite en niveaux de
gris, classification par plus proche voisin (distance L1).

**76,47 % de reconnaissance (26/34)** sur ce premier essai. En comparant les
7 metriques de [`02-parking-distances`](https://github.com/Moha-df/smart-parking-lbp-to-cnn/tree/main/02-parking-distances) (`compare.py`), Chi-2
fait mieux : **88,24 % (30/34)**.

| Metrique | Taux |
| --- | --- |
| L1 | 76,47 % |
| L2 / L2 au carre | 67,65 % |
| Bhattacharyya | 85,29 % |
| **Chi-2** | **88,24 %** |
| Correlation | 82,35 % |
| Intersection | 76,47 % |

### Explications visuelles (`explain_classification.py`)

Pour chaque image test, reconstruit la meme decision que `classify.py`
(distance L1) et genere une image "TEST au-dessus / plus proche voisin du
training en-dessous / verdict" :

```
resultats/classification/bien_classes/  <categorie>__<test>__vs__<voisin>.png
resultats/classification/mal_classes/   <categorie>__<test>__vs__<voisin>.png
```

Ca permet de voir, image par image, grace a (ou a cause de) quel objet du
training le plus-proche-voisin a tranche. Exemple d'echec caracteristique :
une botte UGG confondue avec une boite de jeu de societe (silhouette
elancee similaire une fois reduite a un histogramme LBP global, qui ne
capture pas la structure spatiale de l'objet).

## Prochaine etape

Reprendre l'approche multi-echelle / spatiale de [`04-parking-lbp-multiechelle`](https://github.com/Moha-df/smart-parking-lbp-to-cnn/tree/main/04-parking-lbp-multiechelle) (LBP
par blocs) puis le CNN de [`05-parking-cnn`](https://github.com/Moha-df/smart-parking-lbp-to-cnn/tree/main/05-parking-cnn), sur ces memes images composites.

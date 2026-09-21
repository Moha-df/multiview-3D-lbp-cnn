"""Classification par CNN : chargement des images, architecture, entrainement.

Contrairement a 01/02 (descripteur LBP fait main), le reseau apprend
directement sur les pixels de l'image composite 6-vues (couleur, 768x128).
Le jeu training ne compte que 94 images pour 5 categories, d'ou une
architecture volontairement petite, une legere augmentation de donnees et un
arret anticipe sur une validation decoupee dans le training - exactement la
demarche de parking-old/05-parking-cnn, adaptee au multi-classe et a la
couleur.
"""

import cv2
import numpy as np

# Les vues sont mises a l'echelle par le meme facteur en largeur et en
# hauteur (768x128 -> 384x64) pour ne pas deformer les 6 vues carrees.
IMG_W, IMG_H = 384, 64
N_CLASSES = 5


def load_images(selection, img_w=IMG_W, img_h=IMG_H):
    """[(chemin, label)] -> tenseur (N, img_h, img_w, 3) uint8 et labels."""
    x = np.zeros((len(selection), img_h, img_w, 3), dtype=np.uint8)
    y = np.empty(len(selection), dtype=np.int64)
    for i, (path, label) in enumerate(selection):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("image illisible : %s" % path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x[i] = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        y[i] = label
    return x, y


def stratified_split(x, y, val_fraction, seed):
    """Decoupe (x, y) en deux sous-ensembles, meme proportion de chaque classe."""
    rng = np.random.default_rng(seed)
    idx_train, idx_val = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        idx_val.append(idx[:n_val])
        idx_train.append(idx[n_val:])
    idx_train = np.concatenate(idx_train)
    idx_val = np.concatenate(idx_val)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return (x[idx_train], y[idx_train]), (x[idx_val], y[idx_val])


def build_model(img_w=IMG_W, img_h=IMG_H, n_classes=N_CLASSES, augmentation=True, seed=None):
    """Petit CNN multi-classe : 3 blocs conv, tete dense, sortie softmax."""
    import tensorflow as tf
    from tensorflow.keras import Input, Sequential, layers

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    modele = Sequential(name="objets3d_cnn")
    modele.add(Input(shape=(img_h, img_w, 3)))
    modele.add(layers.Rescaling(1.0 / 255))
    if augmentation:
        modele.add(layers.RandomFlip("horizontal"))
        modele.add(layers.RandomTranslation(0.05, 0.05, fill_mode="constant"))
        modele.add(layers.RandomZoom(0.1))
    for filtres in (16, 32, 64):
        modele.add(layers.Conv2D(filtres, 3, activation="relu", padding="same"))
        modele.add(layers.MaxPooling2D(2))
    modele.add(layers.Flatten())
    modele.add(layers.Dense(64, activation="relu"))
    modele.add(layers.Dropout(0.5))
    modele.add(layers.Dense(n_classes, activation="softmax"))
    modele.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modele


def train_and_evaluate(training, test, seed, img_w=IMG_W, img_h=IMG_H, epochs=40,
                       patience=6, batch_size=16, val_fraction=0.2, verbose=0):
    """Entraine un CNN sur `training`, evalue sur `test` (jeux disjoints).

    Renvoie (taux, historique d'entrainement, modele, y_test, predictions).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.backend.clear_session()
    x_train_full, y_train_full = load_images(training, img_w, img_h)
    x_test, y_test = load_images(test, img_w, img_h)
    (x_train, y_train), (x_val, y_val) = stratified_split(
        x_train_full, y_train_full, val_fraction, seed)

    modele = build_model(img_w, img_h, augmentation=True, seed=seed)
    arret = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    historique = modele.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=[arret], verbose=verbose)

    proba = modele.predict(x_test, batch_size=32, verbose=0)
    predictions = proba.argmax(axis=1)
    taux = 100.0 * float((predictions == y_test).mean())
    return taux, historique.history, modele, y_test, predictions

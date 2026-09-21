"""Rendu de 6 vues (une par face du cube englobant) pour chaque objet 3D.

Chaque objet est charge depuis son zip (maillage + texture reelle, voir
download_dataset.js), centre, mis a l'echelle, puis "photographie"
virtuellement selon les 6 directions cardinales (face, dos, droite, gauche,
dessus, dessous). Les 6 vues sont concatenees horizontalement en une seule
image composite (meme principe que la mosaique RGB de
smart-parking-lbp-to-cnn/03-parking-lbp-couleur), qui devient l'imagette utilisee par le
LBP.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

from dataset import CATEGORIES

VIEW_SIZE = 128


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _pose(eye, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = eye
    return pose


def camera_poses(distance):
    """6 poses camera (face, dos, droite, gauche, dessus, dessous)."""
    return {
        "face": _pose((0, 0, distance), np.eye(3)),
        "dos": _pose((0, 0, -distance), _rot_y(180)),
        "droite": _pose((distance, 0, 0), _rot_y(90)),
        "gauche": _pose((-distance, 0, 0), _rot_y(-90)),
        "dessus": _pose((0, distance, 0), _rot_x(-90)),
        "dessous": _pose((0, -distance, 0), _rot_x(90)),
    }


def load_mesh(obj_path):
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def normalize(mesh):
    """Centre le maillage et le met a l'echelle dans une boite ~[-0.8, 0.8]."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    centre = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()
    mesh.apply_translation(-centre)
    if extent > 0:
        mesh.apply_scale(1.6 / extent)
    return mesh


def render_object(renderer, obj_path):
    mesh = normalize(load_mesh(obj_path))
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    vues = []
    for pose in camera_poses(distance=3.0).values():
        scene = pyrender.Scene(bg_color=[0.6, 0.6, 0.6], ambient_light=[0.4, 0.4, 0.4])
        scene.add(pyrender_mesh)
        scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
        couleur, _ = renderer.render(scene)
        vues.append(Image.fromarray(couleur).resize((VIEW_SIZE, VIEW_SIZE)))

    composite = Image.new("RGB", (VIEW_SIZE * len(vues), VIEW_SIZE))
    for i, vue in enumerate(vues):
        composite.paste(vue, (i * VIEW_SIZE, 0))
    return composite


def extract_and_render(renderer, zip_path, dest_path):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        obj_path = next(Path(tmp).rglob("*.obj"))
        # Les modeles Gazebo Fuel referencent la texture par son seul nom de
        # fichier dans le .mtl (`map_Kd texture.png`) alors qu'elle vit dans
        # materials/textures/ : on la recopie a cote de l'obj pour que le
        # resolveur de trimesh la retrouve.
        for texture in Path(tmp).rglob("*.png"):
            if texture.parent != obj_path.parent:
                shutil.copy(texture, obj_path.parent / texture.name)
        composite = render_object(renderer, obj_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(dest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-root", default="data/renders")
    args = parser.parse_args()

    renderer = pyrender.OffscreenRenderer(VIEW_SIZE, VIEW_SIZE)
    total, echecs = 0, 0
    try:
        for categorie in CATEGORIES:
            for split in ("train", "test"):
                src_dir = Path(args.raw_root) / categorie / split
                if not src_dir.exists():
                    continue
                for zip_path in sorted(src_dir.glob("*.zip")):
                    dest = Path(args.out_root) / categorie / split / (zip_path.stem + ".png")
                    if dest.exists():
                        continue
                    try:
                        extract_and_render(renderer, zip_path, dest)
                        total += 1
                        print("OK    %-28s %-6s %s" % (categorie, split, zip_path.stem))
                    except Exception as exc:
                        echecs += 1
                        print("ECHEC %-28s %-6s %s -> %s" % (categorie, split, zip_path.stem, exc))
    finally:
        renderer.delete()
    print("\n%d images composites generees, %d echecs -> %s" % (total, echecs, args.out_root))


if __name__ == "__main__":
    main()

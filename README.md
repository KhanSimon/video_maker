# Pose Video Demo

Projet Python indépendant pour générer des vidéos de démonstration autour de poses 3D de joueurs de football. Le pipeline ne dépend pas de `field_converter` : il charge ses propres données, configs et modules depuis ce repo.

## Installation

```bash
uv sync
source .venv/bin/activate
```

Les dépendances principales sont OpenCV, NumPy, PyYAML, tqdm, PyVista et VTK.

## Données attendues

La séquence fournie est `ARG_FRA_182345` :

```text
data/
  videos/ARG_FRA_182345.mov
  cameras/ARG_FRA_182345.npz
  boxes/ARG_FRA_182345.npy
  xrel_sam/ARG_FRA_182345.npy
  root_cam/ARG_FRA_182345.npy
  pitch/pitch_points.txt
  teams/ARG_FRA_182345.txt
```

Les caméras doivent contenir `K`, `R`, `t` et idéalement `k`. Les poses sont reconstruites avec `X_cam = X_rel_sam + root_cam[:, :, None, :]`, puis converties en monde par convention row-vector : `X_world = (X_cam - t) @ R`.

## Commandes

Sanity check des données :

```bash
python scripts/sanity_check.py --config configs/sequence_demo.yaml
```

Rendu broadcast avec overlays :

```bash
python scripts/render_broadcast.py --config configs/render_broadcast.yaml
```

Rendu drone / caméra virtuelle 3D :

```bash
python scripts/render_drone.py --config configs/render_drone.yaml
```

Démo complète avec clips intermédiaires puis assemblage :

```bash
python scripts/render_full_demo.py --config configs/sequence_demo.yaml
```

Assembler des clips déjà rendus :

```bash
python scripts/assemble_demo.py --config configs/assemble_demo.yaml
```

Les vidéos sont écrites dans `outputs/clips/` et `outputs/final/`. Quelques PNG de debug sont sauvegardés dans `outputs/frames/`.

## Pipeline

Le rendu broadcast lit la vidéo source avec OpenCV, projette les lignes du terrain et les stickmen dans la caméra réelle, puis dessine boxes, sol vert et overlays selon le YAML.

Le rendu drone convertit les poses en coordonnées monde et utilise PyVista en off-screen pour afficher un terrain vert clair, les lignes rouges et les stickmen 3D. Le mode recommandé est `travel`, avec une pose de caméra de départ et une pose de fin :

```yaml
camera_motion:
  type: travel
  smoothstep: true
  look_at_distance_m: 100.0
  start:
    position:
      mode: spherical
      origin: players_center
      radius_m: 50.0
      azimuth_deg: 270.0
      elevation_deg: 24.0
    look_at: null
    zoom:
      mode: broadcast_camera
      factor: 1.0
  end:
    position:
      mode: cartesian
      xyz: [-18.0, -8.0, 12.0]
    look_at: null
    zoom:
      mode: fov
      fov_deg: 24.0
```

Les points acceptent `mode: cartesian` avec `xyz`, `mode: spherical` avec `origin/radius_m/azimuth_deg/elevation_deg`, ou `mode: target` avec `target` et `offset`. Les origines pratiques sont `world_origin`, `field_center`, `players_center` et `broadcast_camera`. Si `look_at` vaut `null` ou est absent, le drone reprend la direction optique de la camera broadcast : la pose start utilise l'orientation de la frame de debut, la pose end celle de la derniere frame. `look_at_distance_m` ne change pas l'orientation, seulement la distance du point cible place devant la camera virtuelle. Les anciens modes `static_broadcast_like`, `arc_shot`, `arc_descend` et `spiral_descend` restent supportés.

Le zoom du drone est converti en angle de champ vertical PyVista. Par defaut, `zoom.mode: broadcast_camera` calcule ce champ depuis les intrinsics de la vraie camera (`fy` dans `K`) : la pose start utilise la frame de debut du rendu drone, et la pose end utilise la derniere frame. `factor: 2.0` equivaut a un zoom x2 autour de cette reference, donc un champ plus serre. Pour forcer une valeur absolue, utiliser `zoom.mode: fov` avec `fov_deg`.

La démo complète suit une timeline YAML : RGB, lignes terrain, boxes, stickmen, transition vers fond blanc/sol vert, puis caméra drone.

## Notes

`src/pose_video_demo/skeleton/joints.py` contient une topologie Body25-like volontairement simple. Elle doit être vérifiée si le mapping exact des 25 joints diffère.

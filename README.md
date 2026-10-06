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

Les caméras doivent contenir `K`, `R`, `t` et idéalement `k`. Pour le rendu, `root_cam` désigne par défaut la position du pelvis (joint BODY_25 numéro 8). Les poses sont recentrées sur ce joint avant reconstruction : `X_cam = (X_rel_sam - X_rel_sam[:, :, 8:9, :]) + root_cam[:, :, None, :]`, puis converties en monde par convention row-vector : `X_world = (X_cam - t) @ R`. Ce recentrage accepte aussi les poses déjà centrées sur le pelvis et évite un décalage vertical lorsque leur origine est située aux pieds.

`poses.root_joint_index` permet de choisir un autre joint. Si le fichier contient une translation de l'origine de la pose, comme les sorties de `ground_ray`, utiliser `poses.root_joint_index: null` pour conserver l'addition directe. L'outil `build_x_world_estim.py` applique la même convention (pelvis par défaut, ou `--root-joint-index none` pour une translation). Le préparateur de calibration conserve ses translations de l'origine.

## Corps MHR dans les vidéos

Dans `configs/sequence_demo.yaml`, chaque segment de `timeline` et chaque étape de `drone.camera_motion` possède son propre `MHR: true` ou `false`. Le commutateur global a été retiré de cette config. Les paramètres de qualité restent communs :

```yaml
mhr:
  lod: 1                  # résolution du maillage, de 0 à 6
  smooth_shading: true     # éclairage lissé de la surface

paths:
  vertices: data/vertices/ARG_FRA_182345.npy
  mhr_assets: data/mhr
  # Conserver aussi les chemins vidéo, caméras, xrel_sam, root_cam, terrain et équipes.

timeline:
  - name: stickmen
    start_frame: 200
    end_frame: 300
    MHR: false
    show_stickmen: true
  # Conserver les autres segments contigus de la timeline.

drone:
  camera_motion:
    start:
      MHR: false
      # Conserver position, look_at et zoom.
    mid1:
      frame: 512
      hold_frames: 45
      MHR: true
      # Conserver position, look_at et zoom.
    mid2:
      frame: 625
      hold_frames: 79
      MHR: false
    end:
      MHR: true
```

Dans cet exemple, le drone affiche des squelettes jusqu'à la frame 511, des MHR de 512 à 624, puis des squelettes à partir de 625. **Le choix d'une étape prend effet à sa frame d'arrivée**, y compris pendant sa pause, et reste actif jusqu'au prochain choix. `start` s'applique dès la première frame du drone. `end` s'applique uniquement à la dernière frame (`end_frame - 1`) ; pour changer la représentation pendant le trajet final, régler le dernier `mid`. Sur un clip d'une seule frame, seul `start` s'applique.

Une étape sans champ `MHR` conserve le choix précédent ; `start` prend par défaut celui du segment drone de la timeline. Un segment sans champ `MHR` reprend le réglage de sa section `broadcast`/`drone`, puis le défaut `false`. Le commutateur global reste accepté pour les anciennes configs et les rendus séparés (`configs/render_drone.yaml`) ; les valeurs locales ont toujours priorité. Des chaînes comme `"false"` sont refusées : utiliser les booléens YAML sans guillemets.

`MHR: true` remplace les squelettes par les surfaces humaines, avec les couleurs d'équipe, dans le rendu drone, le broadcast et les transitions. Les options `show_stickmen` continuent à commander la visibilité des joueurs indépendamment de leur représentation. Si aucun passage visible n'utilise MHR, aucun fichier de maillage n'est chargé. Les données MHR sont partagées entre les clips et restent en cache lors des changements de représentation. Les changements n'ajoutent aucune frame et ne modifient ni les poses ni le trajet de caméra. Le viewer interactif n'utilise pas cette option.

Les sommets seuls ne donnent pas la connectivité : il faut également les triangles communs à tous les joueurs et à toutes les frames. Les petites tables officielles MHR sont préparées une fois avec :

```bash
python scripts/prepare_mhr_assets.py
```

Elles sont déjà présentes dans `data/mhr` pour cet espace de travail. Ce dossier, comme le reste de `data`, est ignoré par Git ; relancer la commande sur une autre installation. Elle télécharge les topologies et conversions officielles à une révision fixée, conserve leur licence Apache 2.0 et écrit leur provenance dans `data/mhr/sources.json`. Aucun checkpoint SAM, PyTorch ou modèle paramétrique MHR n'est nécessaire au rendu. Les commandes vidéo habituelles restent identiques.

### Signification des trois exports

Les fichiers `ARG_FRA_182345.npy` inspectés ont l'ordre **(frame, joueur, …)** : 1 241 frames et 22 joueurs.

| Export | Dimensions | Interprétation |
| --- | --- | --- |
| `shape` | `(1241, 22, 45)` | Coefficients d'identité de la surface : 20 pour le corps, 20 pour la tête et 5 pour les mains. Ce sont des composantes statistiques, pas 45 mensurations directement interprétables. |
| `body_pose` | `(1241, 22, 133)` | Paramètres compacts de pose produits par SAM 3D Body : angles articulaires et quelques translations internes. Ce ne sont ni 133 joints ni des positions XYZ. |
| `vertices` | `(1241, 22, 18439, 3)` | Positions XYZ des 18 439 sommets du corps déjà déformé, au **LOD 1**, en mètres dans les axes caméra SAM, avant placement global sur le terrain. |

Le modèle MHR sépare la forme de la surface des proportions du squelette : `shape` ne contrôle donc pas à lui seul la longueur des os ou la taille. Une identité est en principe constante pour un joueur ; les valeurs exportées peuvent fluctuer parce qu'elles ont été estimées sur chaque image. Voir la [description officielle des paramètres MHR](https://github.com/facebookresearch/MHR#model-parameters) et la section 3.3 du papier fourni.

Pour les **133 paramètres**, le code SAM 3D Body local permet de préciser la convention : les indices Python `0:124` contiennent des paramètres angulaires en radians, `124:130` six paramètres de translation, et `130:133` trois paramètres de mâchoire. Ces trois derniers sont mis à zéro et écartés par `mhr_forward`, qui utilise `body_pose[..., :130]`. La rotation et la translation globales sont fournies séparément : `3 + 3 + 130 = 136` paramètres de pose utilisés par MHR, auxquels s'ajoutent 68 paramètres de proportions du squelette pour atteindre les 204 du papier. Les poses de mains peuvent aussi être remplacées par une sortie dédiée. Voir les [conversions SAM](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/models/modules/mhr_utils.py) et la [construction des paramètres MHR](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/models/heads/mhr_head.py).

**Pour relire les corps déjà calculés, seuls `vertices` et les triangles sont nécessaires**, en plus des caméras et données de placement déjà utilisées par le projet. `body_pose` et `shape` ne sont pas chargés. Ils seraient utiles pour modifier ou recalculer les corps, mais ces deux tableaux ne suffisent pas à reproduire exactement les sommets : il faudrait aussi le modèle, les paramètres globaux, les proportions du squelette et les sorties mains/expressions correspondantes.

Le rendu applique le même ancrage que les squelettes : `V_cam = V_source - X_rel_sam[pelvis] + root_cam`, puis `V_world = (V_cam - t) @ R`. `poses.root_joint_index: null` conserve la translation directe ; `poses.xrel_axis_signs` s'applique aussi aux sommets. Les exports doivent partager le même repère source et le même ordre de joueurs/frames que les poses. Le chargeur accepte `(T,N,V,3)` et `(N,T,V,3)` ; `mhr.input_layout: tn` ou `nt` lève une éventuelle ambiguïté. Les coordonnées invalides et les corps entièrement nuls sont masqués. Le fichier de sommets d'environ 6 Go est ouvert en lecture mémoire mappée, avec conversion et rendu d'une seule frame à la fois.

### Niveau de détail

| `mhr.lod` | Sommets par joueur | Triangles par joueur | Usage |
| --- | ---: | ---: | --- |
| 0 | 73 639 | 147 274 | Densification interpolée du LOD 1, coûteuse ; n'ajoute pas de détail observé. |
| 1 | 18 439 | 36 874 | Export original, détail maximal disponible dans les données. |
| 2 | 10 661 | 21 318 | Réduction modérée. |
| 3 | 4 899 | 9 794 | Compromis pour les plans larges. |
| 4 | 2 461 | 4 918 | Rendu plus léger. |
| 5 | 971 | 1 938 | Silhouette simplifiée. |
| 6 | 595 | 1 186 | Résolution minimale. |

La conversion utilise les [correspondances barycentriques officielles](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_LOD_conversion) et la topologie de chaque LOD. Ces correspondances restent identiques entre les frames. Elle fonctionne directement depuis les sommets LOD 1, sans réestimer les paramètres du corps. `smooth_shading` règle uniquement l'éclairage lissé, sans modifier le nombre de sommets. Les exports contiennent une géométrie humaine, sans texture de maillot ; le rendu emploie donc une couleur par équipe.

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

Construire les poses estimees en coordonnees monde :

```bash
python scripts/build_x_world_estim.py --sequence ARG_FRA_182345 --overwrite
```

Par defaut, la sortie est ecrite dans `data/x_world_estim/<sequence>.npy` avec le layout `(N,T,25,3)`, qui est aussi la convention de `data/x_world_gt`.

### Lisser les positions des joueurs avant le rendu

Le rendu de `sequence_demo.yaml` relit directement `paths.root_cam` à chaque lancement, puis reconstruit les poses depuis `xrel_sam`, `root_cam` et les caméras. Il ne lit pas `data/x_world_estim` : reconstruire ce fichier n'est donc pas nécessaire pour les vidéos. Une vidéo déjà générée conserve ses anciennes données.

Le lissage est une commande de préparation indépendante du rendu :

```bash
python scripts/smooth_root_cam.py --sequence ARG_FRA_182345 --window-frames 15 --in-place
```

Cette commande remplace `data/root_cam/ARG_FRA_182345.npy` après en avoir sauvegardé une copie exacte dans `data/root_cam/backups/`, avec un nom daté. Le YAML existant utilisera donc le fichier lissé au prochain rendu. Un rapport `ARG_FRA_182345.smoothing.json` contient les chemins, les empreintes des fichiers, les paramètres et les mesures avant/après. Le script refuse de lisser à nouveau un fichier déjà traité ; pour modifier le réglage, repartir de la sauvegarde originale :

```bash
python scripts/smooth_root_cam.py --sequence ARG_FRA_182345 \
  --input 'data/root_cam/backups/ARG_FRA_182345.<date>.npy' \
  --output data/root_cam/ARG_FRA_182345.npy --overwrite --window-frames 21
```

Remplacer `<date>` par le nom indiqué dans le rapport. Si un nouveau fichier brut remplace le `root_cam`, la commande `--in-place` peut de nouveau être utilisée ; une nouvelle sauvegarde est créée. Pour calculer les mesures sans écrire de fichier, ajouter `--dry-run`. Sans `--in-place` ni `--output`, la sortie est séparée, dans `data/root_cam_smoothed/<sequence>.npy`, et le YAML n'est pas modifié.

Le filtre Savitzky–Golay ajuste localement un polynôme quadratique sur une fenêtre centrée de `window_frames` images (nombre impair, au moins 5). Une fenêtre plus grande augmente le lissage et peut atténuer les changements de direction rapides ; 15 est le réglage initial. La conversion `root_world = (root_cam - t) @ R` sépare le déplacement du joueur de celui de la caméra broadcast ; les positions filtrées sont ensuite reconverties dans le repère caméra. Le calcul utilise uniquement NumPy et ne dépend pas des FPS.

Chaque joueur et chaque portion valide sont filtrés séparément. Les frames manquantes (NaN, infinis ou racines entièrement nulles) sont conservées, sans interpolation à travers les trous. Les portions de moins de 5 frames restent inchangées. Le layout `(N,T,3)` ou `(T,N,3)`, le dtype et le nombre de frames sont préservés. Ce traitement lisse la translation globale ; il ne modifie pas les articulations relatives de `xrel_sam` ni les sommets MHR. Le rendu ne lance jamais ce filtre automatiquement.

### Importer une calibration externe

Le preparateur accepte les poses et translations en `(T,N,...)` ou `(N,T,...)`, convertit le JSON de calibration vers le format camera du projet et cree les coordonnees monde. Le terrain peut etre complet (`pitch.mode: standard`) ou reconstruit uniquement depuis les annotations `calibration.input_lines` du JSON (`pitch.mode: calibration_lines`).

```bash
python scripts/prepare_calibrated_sequence.py \
  --config configs/prepare_gardien_rennes.yaml \
  --overwrite
```

Pour `gardien_rennes`, les sorties sont :

```text
data/cameras/gardien_rennes.npz
data/pitch/gardien_rennes.txt
data/teams/gardien_rennes.txt
data/x_world_estim/gardien_rennes.npy
```

Avec `translation.method: input`, le fichier `t_per_player` contient un vecteur camera `(x,y,z)` par joueur et par frame et joue le role de `root_cam`. Le JSON utilise un autre repere terrain : `camera.source_to_world_axis_signs: [1, -1, -1]` inverse son axe lateral et place Z vers le haut tout en conservant une rotation camera propre. `poses.xrel_axis_signs` permet de corriger independamment une source de poses dont un axe relatif serait inverse.

### Translation geometrique par rayon-sol

La variante `ground_ray` n'utilise pas `t_per_player`. Pour chaque joueur et chaque frame, elle selectionne le joint relatif le plus bas, prend le pixel du meme joint dans `x_2d`, intersecte son rayon camera avec le plan du sol, puis calcule la translation qui place ce joint sur l'intersection. L'origine de la pose relative peut donc etre au bassin, aux pieds ou ailleurs.

```bash
python scripts/prepare_calibrated_sequence.py \
  --config configs/prepare_gardien_rennes_ground_ray.yaml \
  --overwrite
```

Les nouvelles estimations sont isolees dans :

```text
data/geometric_estim/gardien_rennes/x_world.npy
data/geometric_estim/gardien_rennes/root_cam.npy
data/geometric_estim/gardien_rennes/diagnostics.npz
```

Par defaut, la hauteur relative est evaluee apres rotation des offsets vers le monde (`contact_selection: world_z`). Pour une source particuliere, `poses.xrel_axis_signs` corrige les signes de X/Y/Z, tandis que `contact_selection: relative_axis` permet de choisir explicitement l'axe vertical et son sens. Le filtre temporel combine une mediane glissante, un passe-bas avant/arriere et une limite de vitesse; il travaille dans le monde pour rester independant d'un changement de focale ou d'orientation camera.

Pour une zone de jeu partielle, `pitch.floor_extent_m: [x_min, x_max, y_min, y_max]` definit uniquement l'emprise du sol. Pour `gardien_rennes`, les lignes viennent du JSON et le sol correspond au demi-terrain cote but gauche; aucune ligne mediane ni aucun rond central absent de la calibration n'est invente.

Une camera fixe produit une camera identique sur toutes les frames. Pour une camera mobile ou un zoom variable, `inputs.calibration_json` peut etre une liste de JSON. Des champs `frame_index` dans les calibrations definissent leurs positions temporelles; sans index, les calibrations sont reparties uniformement et interpolees.

Visualiser le resultat sur le nouveau terrain :

```bash
python scripts/interactive_viewer.py \
  --config configs/interactive_viewer_gardien_rennes.yaml
```

Visualisateur interactif Open3D :

```bash
python scripts/interactive_viewer.py --config configs/interactive_viewer.yaml
```

La configuration principale charge directement `data/sequence/ekstraklasa_001999.npz`. Le mode `source: npz` lit `joints_pred_world_m`, applique `valid_mask` pour masquer les joueurs absents et utilise `frame_numbers` pour afficher le numero de frame source. Les cles sont configurables avec `npz_pose_key`, `npz_valid_mask_key` et `npz_frame_numbers_key`.

Ce NPZ suffit pour afficher et animer les poses dans le repere monde. Deux informations ne sont pas presentes : les equipes reelles et le FPS (ou des timestamps). Sans fichier `paths.teams`, les joueurs utilisent la couleur de repli; la vitesse de lecture vient de `sequence.fps`, regle ici a 50. Les 50 `pitch_points_world_m` ne donnent pas la connectivite des lignes. Le viewer utilise donc `data/pitch/pitch_points.txt`, dont les bornes correspondent exactement au terrain du NPZ.

Pour les anciennes sources `estim` qui ne fournissent que les poses relatives et les translations camera, creer d'abord les points 3D avec :

```bash
python scripts/build_x_world_estim.py  --sequence ARG_FRA_182345 --overwrite
```

La config permet de choisir `interactive_viewer.source: gt`, `estim` ou `both`. En mode `both`, la GT est rouge et l'estimation garde les couleurs d'equipe. `M` alterne entre les modes `gt`, `estim` et `both`. Controles : avec `mouse_look: true`, mouvement souris pour regarder autour en mode FPS; avec `mouse_look: false`, Open3D reprend le drag/orbit natif. Scroll pour zoomer, `Espace` play/pause, fleches gauche/droite pour changer de frame, `Z/S/Q/D` pour avancer/reculer/gauche/droite, `A/E` pour monter/descendre, `+/-` pour la vitesse temporelle entre `0.25x` et `2x`, `R` reset camera, `Echap` quitter.

Dans `configs/interactive_viewer.yaml`, `floor_color_mode: fixed` garde la couleur du terrain constante, tandis que `floor_color_mode: lit` laisse l'eclairage Open3D influencer la couleur.

Le filtre global des joueurs se configure directement sous `interactive_viewer` :

```yaml
interactive_viewer:
  player_indices: [0, 4, 12]
```

Seuls ces joueurs sont alors affiches dans la scene principale, le contexte temporel et la minimap. `player_indices: []` affiche tous les joueurs. Ce filtre global est distinct de `temporal_context.player_indices`, qui choisit les joueurs possedant des poses intermediaires mais ne peut pas rendre visible un joueur masque globalement.

Le contexte temporel affiche, pour plusieurs joueurs, les poses situees aux offsets demandes ainsi que les frames intermediaires :

```yaml
temporal_context:
  enabled: true
  player_indices: [0, 4, 12]
  frame_offsets: [-100, 100, 250]
  show_intermediate: true
  intermediate_stride: 1
  intermediate_pastel_coefficient: 0.55
  endpoint_opacity: 0.85
```

Les offsets sont relatifs a l'index courant et peuvent etre positifs ou negatifs. Chaque pose intermediaire reprend la couleur de base de sa couche, rouge pour la GT ou couleur d'equipe pour l'estimation, puis l'eclaircit vers le blanc avec `intermediate_pastel_coefficient`. `0.0` conserve la couleur d'origine et `1.0` produit du blanc. Pour les offsets explicitement listes, `endpoint_opacity: 0.0` utilise exactement cette couleur intermediaire et `endpoint_opacity: 1.0` utilise la couleur du joueur a la frame principale; les valeurs intermediaires melangent les deux. `intermediate_stride: 1` affiche chaque frame; une valeur plus grande allege la scene. Les indices hors de la sequence et les poses marquees invalides sont ignores sans bouclage.

La minimap reste fixee en haut a droite de la fenetre et suit la frame courante :

```yaml
minimap:
  enabled: true
  source_layer: auto
  width_px: 320
  height_px: 215
  margin_px: 24
  padding_px: 12
  panel_color: [38, 92, 58]
  pitch_line_color: [245, 248, 246]
  pitch_line_width_px: 1.25
  marker_radius_px: 5.5
  position_joint_index: 8
  clip_positions: true
```

`enabled: false` la masque. Avec `source_layer: auto`, elle suit la couche `gt` ou `estim` affichee et choisit `estim` en mode `both`; une couche peut aussi etre forcee explicitement. Les positions viennent du joint configure, tandis que les marqueurs gardent les couleurs d'equipe meme lorsque les squelettes GT sont rouges. Les joueurs invalides a la frame courante sont omis.

La position/orientation initiale du visualisateur se regle avec :

```yaml
initial_camera:
  position: [0.0, -85.0, 25.0]
  look_at: [0.0, 0.0, 1.0]
  up: [0.0, 0.0, 1.0]
```

Cette camera est aussi reappliquee avec `R`.

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

### Temporalité des vidéos : uniquement des frames

Tous les repères sont des **numéros absolus de frames dans la séquence source**, numérotés à partir de 0. Une plage `start_frame: 400`, `end_frame: 850` rend les frames **400 à 849**, soit 450 images. La borne `end_frame` est toujours exclue. Il n'y a aucun décalage implicite ni conversion par les FPS.

Dans `configs/sequence_demo.yaml`, la `timeline` est l'unique définition des plages. Les sections `broadcast` et `drone` règlent leur apparence et leur caméra. Pour les rendus séparés, `configs/render_broadcast.yaml` et `configs/render_drone.yaml` définissent leur plage directement dans la section correspondante. Sans plage ni timeline, le rendu couvre toute la source disponible. Avec le YAML de démo, un rendu séparé reprend les plages contiguës de son renderer dans la timeline.

```yaml
timeline:
  - name: transition_white_green
    start_frame: 300
    end_frame: 400
    transition:
      rgb_to_white: true
      boxes_fade_out: true
    show_pitch_lines: true
    show_stickmen: true
  - name: drone
    start_frame: 400
    end_frame: 850
    renderer: drone
    show_green_floor: true
    show_pitch_lines: true
    show_stickmen: true
```

Les fondus occupent les frames du segment `transition`. La dernière image broadcast est ici la frame 399 et la première image drone la frame 400. Chaque segment doit commencer exactement à la fin du précédent. Chevauchements, trous, noms dupliqués, bornes non entières et pauses incompatibles sont signalés avant de générer les clips. L'assemblage concatène les images sans rejouer de frames pour les transitions ou les pauses.

Le trajet accepte autant d'étapes intermédiaires que nécessaire avec `mid1`, `mid2`, `mid3`, etc., sous `drone.camera_motion`, au même niveau que `start` et `end` :

```yaml
drone:
  green_floor_start_frame: 425
  camera_motion:
    type: travel
    smoothstep: true
    # Conserver les réglages spatiaux start et end.
    mid1:
      frame: 512
      hold_frames: 45
      position: {mode: cartesian, xyz: [-40, -20, 10]}
      look_at: {mode: cartesian, xyz: [40, 0, 2]}
      zoom: {mode: broadcast_camera, factor: 1.0}
    mid2:
      frame: 625
      hold_frames: 4
      position: {mode: cartesian, xyz: [10, -10, 2]}
      look_at: {mode: cartesian, xyz: [50, -80, 2]}
      zoom: {mode: factor, factor: 0.5}
    mid3:
      frame: 750
      hold_frames: 10
      position: {mode: cartesian, xyz: [0, -12, 5]}
      look_at: {mode: cartesian, xyz: [50, -40, 2]}
      zoom: {mode: factor, factor: 0.6}
```

`frame` est obligatoire et désigne l'arrivée à l'étape. `hold_frames` est le nombre d'images où la caméra conserve cette pose, **frame d'arrivée incluse** : à la frame 512 avec une pause de 45 frames, la caméra reste fixe de 512 à 556, puis reprend son mouvement à 557. Une valeur de 0, ou l'absence du champ, n'ajoute aucune pause ; la caméra passe tout de même exactement par la pose à la frame indiquée. Les joueurs continuent d'avancer pendant les pauses de caméra.

Le parcours suit l'ordre numérique `start → mid1 → mid2 → mid3 → end`, indépendamment de l'ordre des clés YAML. `start` correspond à `start_frame` et `end` à `end_frame - 1` ; sur un clip d'une seule image, la pose `start` est utilisée. Chaque arrivée doit suivre la dernière frame de pause précédente, et la dernière pause doit finir avant la frame de la caméra `end`.

Chaque étape accepte les mêmes modes de position, cible `look_at` et zoom. `anchor_frame: start` ou `end` choisit la frame de référence spatiale pour les origines `players_center` et `broadcast_camera` (`end` par défaut) ; cela ne change pas la frame d'arrivée. Si une position, une cible ou un zoom est omis, sa valeur est interpolée entre les réglages de début et de fin. Avec `smoothstep: true`, chaque portion du trajet démarre et finit progressivement. Le nom `mid` (ainsi que `middle` et `waypoint`) reste utilisable seul, avec `frame` et `hold_frames` ; pour ajouter d'autres étapes, le renommer en `mid1`.

`green_floor_start_frame` indique la première frame source avec sol vert dans le drone ; `null` ou son absence l'affiche dès le début du clip. `render.debug_frames` utilise aussi les numéros source, dans le broadcast, les transitions et le drone ; les noms des PNG reprennent ces numéros.

`render.output_fps` fixe seulement la cadence de lecture du fichier vidéo. Le changer conserve les mêmes images, étapes, fondus et nombres de frames de pause ; il change la vitesse de lecture. Ce champ peut être omis pour reprendre la cadence de la vidéo source.

Les anciens champs `timeline.start/end`, `arrive_at`, `depart_at`, `hold_duration_s`, `hold_duration_progress` et `green_floor_start_delay_s` sont refusés avec un message de migration. Les réglages redondants `sequence.start_frame`, `sequence.num_frames` et `sequence.fps` ont été retirés des configs vidéo. L'interactive viewer conserve ses propres réglages.

Après modification de la timeline, relancer `scripts/render_full_demo.py` pour régénérer les clips et le film final ; réassembler d'anciens clips conserve leur ancien découpage.

## Notes

`src/pose_video_demo/skeleton/joints.py` contient une topologie Body25-like volontairement simple. Elle doit être vérifiée si le mapping exact des 25 joints diffère.

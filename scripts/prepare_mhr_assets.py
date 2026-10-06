"""Download the small official MHR topology and LOD tables (no model weights)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOPLY import vtkPLYReader

from _bootstrap import add_src_to_path

add_src_to_path()

from video_maker.io.mhr import LOD_VERTEX_COUNTS
from video_maker.utils.paths import resolve_path

# Keep triangle ordering and barycentric mappings from the same upstream revision.
MHR_REVISION = "e412e12c9d7287a598f00edf19242b476b440211"
MHR_BASE_URL = f"https://raw.githubusercontent.com/facebookresearch/MHR/{MHR_REVISION}"


def prepare_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        "LICENSE",
        "tools/mhr_smpl_conversion/assets/mhr_face_mask.ply",
        "web-viewer/data/lod_topology.npz",
        *(f"tools/mhr_LOD_conversion/lod1_to_lod{lod}_mapping.npz" for lod in (0, 2, 3, 4, 5, 6)),
    ]
    manifest = {"repository": "https://github.com/facebookresearch/MHR", "revision": MHR_REVISION, "files": {}}
    for source in sources:
        path = output_dir / Path(source).name
        url = f"{MHR_BASE_URL}/{source}"
        print(f"Downloading {path.name}", flush=True)
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        manifest["files"][path.name] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}

    reader = vtkPLYReader()
    reader.SetFileName(str(output_dir / "mhr_face_mask.ply"))
    reader.Update()
    mesh = reader.GetOutput()
    polygons = mesh.GetPolys()
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    indices = vtk_to_numpy(polygons.GetConnectivityArray())
    if mesh.GetNumberOfPoints() != LOD_VERTEX_COUNTS[1] or indices.size != 36874 * 3:
        raise ValueError("Unexpected official MHR LOD 1 mesh")
    if not np.all(np.diff(offsets) == 3):
        raise ValueError("Expected triangular MHR LOD 1 topology")
    np.save(output_dir / "faces_lod1.npy", indices.reshape(-1, 3).astype(np.int32))
    with np.load(output_dir / "lod_topology.npz", allow_pickle=False) as topology:
        for lod in (0, 2, 3, 4, 5, 6):
            np.save(output_dir / f"faces_lod{lod}.npy", topology[f"lod{lod}_faces"].astype(np.int32))
    (output_dir / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MHR LOD 0..6 ready in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/mhr")
    args = parser.parse_args()
    prepare_assets(resolve_path(args.output_dir))


if __name__ == "__main__":
    main()

"""Figure 2: Braak atlas (top) and per-ROI |d_k| of a DBPS model (bottom) on the fsaverage surface.

Reads a per-ROI CSV written by scripts.braak_alignment (columns roi, abs_d)
and draws lateral/medial views of both hemispheres. Requires nibabel,
nilearn and mne (the fsaverage surface and aparc annotation are downloaded
on first use).

usage:
  python -m scripts.make_braak_figure --per_roi results/braak_union/per_roi_endpoint_shift.csv \
      --out figures/figure2_braak.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import nibabel
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting

from evaluation.braak import braak_stage

PANELS = (("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral"))


def setup_atlas():
    fsav = datasets.fetch_surf_fsaverage()
    fsav_dir = mne.datasets.fetch_fsaverage(verbose=False)
    annots = {}
    for hemi in ("lh", "rh"):
        labels, _, names = nibabel.freesurfer.read_annot(os.path.join(fsav_dir, "label", f"{hemi}.aparc.annot"))
        annots[hemi] = {"labels": labels, "names": [n.decode() if isinstance(n, bytes) else n for n in names]}
    return fsav, annots


def surface_values(roi_to_value: dict, annot: dict, hemi: str) -> np.ndarray:
    values = np.full(len(annot["labels"]), np.nan, dtype=np.float64)
    for idx, name in enumerate(annot["names"]):
        if name in ("unknown", "corpuscallosum"):
            continue
        val = roi_to_value.get(f"ctx-{hemi}-{name}")
        if val is not None:
            values[annot["labels"] == idx] = val
    return values


def plot_two_rows(fsav, annots, rows, out_path: Path, title: str | None):
    """rows: list of (data dict, cmap, vmin, vmax, row label, colorbar label, ticks)."""
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.12], hspace=0.05, wspace=0.0)
    pial = {"left": fsav.pial_left, "right": fsav.pial_right}
    sulc = {"left": fsav.sulc_left, "right": fsav.sulc_right}
    for r, (data, cmap, vmin, vmax, label, cbar_label, ticks) in enumerate(rows):
        for c, (hemi, view) in enumerate(PANELS):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            h = "lh" if hemi == "left" else "rh"
            plotting.plot_surf_stat_map(surf_mesh=pial[hemi], stat_map=surface_values(data, annots[h], h)[:10242],
                                        hemi=hemi, view=view, bg_map=sulc[hemi], cmap=cmap, vmax=vmax,
                                        symmetric_cbar=False, colorbar=False, axes=ax)
            if r == 0:
                ax.set_title(f"{hemi.title()} {view}", fontsize=10)
            if c == 0:
                ax.text2D(-0.10, 0.5, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
                          rotation=90, va="center")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=fig.add_subplot(gs[r, 4]))
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.set_label(cbar_label, fontsize=10)
    if title:
        fig.suptitle(title, fontsize=13, y=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_roi", required=True, help="per-ROI CSV from scripts.braak_alignment")
    ap.add_argument("--out", default="figures/figure2_braak.png")
    ap.add_argument("--label", default="(b) DBPS")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fsav, annots = setup_atlas()
    atlas = {f"ctx-{h}-{n}": braak_stage(f"ctx-{h}-{n}")
             for h in ("lh", "rh") for n in annots[h]["names"] if n not in ("unknown", "corpuscallosum")}
    df = pd.read_csv(args.per_roi)
    model = dict(zip(df.roi, df.abs_d))
    stage_cmap = ListedColormap(["#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"])
    plot_two_rows(fsav, annots, [
        (atlas, stage_cmap, 2, 6, "(a) Braak\natlas", "Braak stage", [2, 3, 4, 5, 6]),
        (model, "hot", 0, float(df.abs_d.max()), args.label, "|Cohen's d|", None),
    ], Path(args.out), args.title)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

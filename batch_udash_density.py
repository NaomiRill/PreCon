"""Batch plotting helper that turns UDASH casts into density maps.

This script is deliberately focused on the mapping workflow. For every
configured UDASH text file it

1. runs the TEOS-10 density calculations by delegating to
   :func:`udash_density.load_udash_file`, so the logic stays consistent
   with the notebook helper; and
2. reloads the year-prefixed TEOS-10 export that ``udash_density`` writes
   into ``Density/`` so the longitude/latitude values used for mapping are
   exactly the ones stored on disk; and
3. scatters all depth levels on a simple longitude/latitude map with point
   colours representing in-situ density, saving one PNG per profile.

The goal is to generate quick-look “physical density” maps for each CTD
probe that can be stacked into a yearly atlas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import udash_density

# I list the UDASH profile files I want to process. New years can be appended
# to this tuple before rerunning the script.
PROFILE_FILES: Sequence[Path] = (
    udash_density.PROFILE_TEXT_FILE,
)

# All figures land in this directory so the notebook can browse them easily.
MAP_OUTPUT_DIR = Path("Density_maps")


def _ensure_sequence(paths: Sequence[Path | str]) -> Iterable[Path]:
    """Normalise any iterable of strings/Paths into Path objects."""

    for entry in paths:
        yield Path(entry)


def _load_teos10_export(profile_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run TEOS-10, reload the on-disk export, and hand everything back."""

    processed = udash_density.load_udash_file(profile_path, save_output=True)
    export_path_str = processed.attrs.get("teos10_output_path")
    if not export_path_str:
        raise RuntimeError(
            f"No TEOS-10 export path recorded for {profile_path}. Did the helper save output?"
        )

    export_path = Path(export_path_str)
    export_df = pd.read_csv(export_path, sep="\t")
    # Tag the DataFrame with the profile name for reference when plotting.
    export_df["source_file"] = profile_path.name
    return processed, export_df, export_path


def _year_from_export_path(export_path: Path) -> str:
    """Extract the leading year token from ``Density/<year>_*.txt`` filenames."""

    stem = export_path.stem
    # Files are named like ``1980_ArcticOcean_phys_oce_1980_teos10``.
    return stem.split("_", 1)[0]


def plot_profile_density_map(
    export_df: pd.DataFrame,
    profile_name: str,
    year: str,
    *,
    output_dir: Path = MAP_OUTPUT_DIR,
) -> Path:
    """Scatter every depth level on a lon/lat map coloured by density."""

    required_columns = {
        udash_density.UDASH_COLUMNS["longitude_deg"],
        udash_density.UDASH_COLUMNS["latitude_deg"],
        "Density_kg_m3",
    }
    missing = required_columns.difference(export_df.columns)
    if missing:
        raise KeyError(
            "Missing columns in TEOS-10 export needed for mapping: "
            + ", ".join(sorted(missing))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{year}_{Path(profile_name).stem}_density_map.png"

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    scatter = ax.scatter(
        export_df[udash_density.UDASH_COLUMNS["longitude_deg"]],
        export_df[udash_density.UDASH_COLUMNS["latitude_deg"]],
        c=export_df["Density_kg_m3"],
        s=12,
        cmap="viridis",
        edgecolors="none",
    )
    ax.set_title(f"Physical density map for {profile_name} ({year})")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Density [kg/m³]")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return figure_path


def process_and_map_profiles(
    paths: Sequence[Path | str] = PROFILE_FILES,
    *,
    output_dir: Path = MAP_OUTPUT_DIR,
) -> list[tuple[str, Path, Path]]:
    """Create one density map per profile and report the file locations."""

    outputs: list[tuple[str, Path, Path]] = []
    for path in _ensure_sequence(paths):
        processed, export_df, export_path = _load_teos10_export(path)
        year = _year_from_export_path(export_path)
        figure_path = plot_profile_density_map(
            export_df,
            profile_name=path.name,
            year=year,
            output_dir=output_dir,
        )
        outputs.append((path.name, export_path, figure_path))
    return outputs


if __name__ == "__main__":
    results = process_and_map_profiles()
    if not results:
        print("No profiles configured. Edit PROFILE_FILES before running.")
    else:
        print("Created density maps for the following profiles:")
        print()
        for profile_name, export_path, figure_path in results:
            print(f"- {profile_name}")
            print(f"  TEOS-10 export: {export_path}")
            print(f"  Density map:   {figure_path}")

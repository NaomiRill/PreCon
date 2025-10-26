"""I wrote this helper so I can load UDASH profiles, run TEOS-10, and plot them.

The idea is that future me (or anyone else working on the thesis) can scan the
comments and instantly remember what every single line is doing. Nothing fancy,
just honest notes from the person who is trying to understand the workflow."""

from __future__ import annotations  # I turn on postponed annotations so type hints stay simple.

# Dataclasses let me bundle profile metadata into something with attributes.
from dataclasses import dataclass
# Path objects make file handling nicer than juggling raw strings everywhere.
from pathlib import Path
# Typing helpers keep the function signatures readable in notebooks.
from typing import Iterable, Optional, Sequence

# I pop warnings whenever the raw data looks suspicious (like negative salinity).
import warnings

# NumPy handles arrays, pandas reads the tab file, and gsw gives me TEOS-10 tools.
import numpy as np
import pandas as pd
import gsw

# I only want to type the file name once, right here at the top.
PROFILE_TEXT_FILE = Path("UDASH/ArcticOcean_phys_oce_1980.txt")

# Whenever something else needs this path I just point it here.
DEFAULT_UDASH_FILE = PROFILE_TEXT_FILE

# Listing the public names helps me remember what to import in the notebook.
__all__ = [
    "UDASH_COLUMNS",
    "UDASHProfileMetadata",
    "PROFILE_TEXT_FILE",
    "DEFAULT_UDASH_FILE",
    "load_udash_file",
    "load_default_profile",
    "iter_profile_metadata",
    "summarize_profiles",
    "preview_profile_metadata",
    "preview_level_samples",
    "plot_teos10_diagnostics",
    "save_teos10_variables",
]

# I store the UDASH column names in one place so I do not keep retyping them.
UDASH_COLUMNS = {
    "profile": "Prof_no",
    "cruise": "Cruise",
    "station": "Station",
    "platform": "Platform",
    "instrument_type": "Type",
    "timestamp": "yyyy-mm-ddThh:mm",
    "longitude_deg": "Longitude_[deg]",
    "latitude_deg": "Latitude_[deg]",
    "pressure_dbar": "Pressure_[dbar]",
    "depth_m": "Depth_[m]",
    "temperature_degC": "Temp_[°C]",
    "salinity_psu": "Salinity_[psu]",
    "quality_flag": "QF",
}


@dataclass
class UDASHProfileMetadata:
    """Tiny container so I can iterate over unique profile info with attributes."""

    profile: int
    cruise: str
    station: str
    platform: str
    instrument_type: str
    timestamp: str
    longitude_deg: float
    latitude_deg: float


def _infer_profile_year(df: pd.DataFrame) -> str:
    """I grab the year from the timestamp so saved files land in the right folder."""

    timestamp_column = UDASH_COLUMNS["timestamp"]
    if timestamp_column not in df.columns:
        return "unknown"  # Sometimes the file just does not include timestamps.

    timestamps = df[timestamp_column].dropna().astype(str)
    if timestamps.empty:
        return "unknown"  # No data means I cannot guess anything.

    first_value = timestamps.iloc[0]
    try:
        year = pd.to_datetime(first_value).year
    except (TypeError, ValueError):
        year = str(first_value)[:4]  # If parsing fails I fall back to slicing.
    return str(year)


def save_teos10_variables(
    df: pd.DataFrame,
    *,
    source_path: Path,
    output_root: Path = Path("Density"),
) -> Path:
    """I write the TEOS-10 results to Density/<year>/... so mapping is easy later."""

    year = _infer_profile_year(df)
    target_dir = output_root / f"{year}"
    target_dir.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists.

    output_path = target_dir / f"{source_path.stem}_teos10.txt"

    export_columns = [
        UDASH_COLUMNS["longitude_deg"],
        UDASH_COLUMNS["latitude_deg"],
        UDASH_COLUMNS["pressure_dbar"],
        UDASH_COLUMNS["depth_m"],
        "Absolute_Salinity_g_kg",
        "Conservative_Temp_degC",
        "Density_kg_m3",
    ]
    available_columns = [column for column in export_columns if column in df.columns]

    df[available_columns].to_csv(
        output_path,
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    return output_path


def load_udash_file(
    path: str | Path,
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
    save_output: bool = True,
    output_root: Path = Path("Density"),
) -> pd.DataFrame:
    """I read the UDASH text file, run TEOS-10, and optionally save the results."""

    path = Path(path)  # Converting early lets me reuse .name and friends.
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        na_values=[missing_value],
        engine="python",
    )

    salinity_column = UDASH_COLUMNS["salinity_psu"]
    if salinity_column in df.columns:
        # Coerce weird strings to NaN so later steps do not explode.
        salinity = pd.to_numeric(df[salinity_column], errors="coerce")
        negative_mask = salinity < 0
        if negative_mask.any():
            warnings.warn(
                "Negative practical salinity values found; treating them as missing.",
                RuntimeWarning,
                stacklevel=2,
            )
            salinity = salinity.mask(negative_mask)
        df[salinity_column] = salinity

    required = [
        salinity_column,
        UDASH_COLUMNS["temperature_degC"],
        UDASH_COLUMNS["pressure_dbar"],
        UDASH_COLUMNS["longitude_deg"],
        UDASH_COLUMNS["latitude_deg"],
    ]
    if drop_na:
        df = df.dropna(subset=required)

    salinity = df[salinity_column].to_numpy(dtype=float)
    pressure = df[UDASH_COLUMNS["pressure_dbar"]].to_numpy(dtype=float)
    longitude = df[UDASH_COLUMNS["longitude_deg"]].to_numpy(dtype=float)
    latitude = df[UDASH_COLUMNS["latitude_deg"]].to_numpy(dtype=float)
    temperature = df[UDASH_COLUMNS["temperature_degC"]].to_numpy(dtype=float)

    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(SA, temperature, pressure)
    rho = gsw.rho(SA, CT, pressure)

    df = df.copy()  # I copy so the original data frame passed in stays untouched.
    df["Absolute_Salinity_g_kg"] = SA
    df["Conservative_Temp_degC"] = CT
    df["Density_kg_m3"] = rho
    df.insert(0, "source_file", path.name)

    if save_output:
        output_path = save_teos10_variables(
            df,
            source_path=path,
            output_root=output_root,
        )
        df.attrs["teos10_output_path"] = str(output_path)
    return df


def load_default_profile(
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
    save_output: bool = True,
    output_root: Path = Path("Density"),
) -> pd.DataFrame:
    """Shortcut that always loads the profile referenced by PROFILE_TEXT_FILE."""

    return load_udash_file(
        PROFILE_TEXT_FILE,
        missing_value=missing_value,
        drop_na=drop_na,
        save_output=save_output,
        output_root=output_root,
    )


def preview_profile_metadata(
    df: pd.DataFrame,
    *,
    max_rows: Optional[int] = 10,
) -> pd.DataFrame:
    """I grab the unique profile info so I can sanity check the file quickly."""

    columns = [
        "source_file",
        UDASH_COLUMNS["profile"],
        UDASH_COLUMNS["cruise"],
        UDASH_COLUMNS["station"],
        UDASH_COLUMNS["platform"],
        UDASH_COLUMNS["instrument_type"],
    ]
    preview = df.loc[:, columns].drop_duplicates().reset_index(drop=True)
    if max_rows is not None:
        preview = preview.head(max_rows)
    return preview


def preview_level_samples(
    df: pd.DataFrame,
    *,
    samples: int = 12,
) -> pd.DataFrame:
    """Instead of printing everything I just peek at evenly spaced rows."""

    if samples <= 0 or df.empty:
        return df.head(0)

    if samples >= len(df):
        return df

    indices = np.linspace(0, len(df) - 1, samples)
    unique_indices = np.unique(indices.round().astype(int))
    return df.iloc[unique_indices]


def iter_profile_metadata(df: pd.DataFrame) -> Iterable[UDASHProfileMetadata]:
    """I yield one UDASHProfileMetadata per unique profile in the DataFrame."""

    seen = set()
    for _, row in df.iterrows():
        key = (row[UDASH_COLUMNS["profile"]], row["source_file"])
        if key in seen:
            continue
        seen.add(key)
        yield UDASHProfileMetadata(
            profile=int(row[UDASH_COLUMNS["profile"]]),
            cruise=str(row[UDASH_COLUMNS["cruise"]]),
            station=str(row[UDASH_COLUMNS["station"]]),
            platform=str(row[UDASH_COLUMNS["platform"]]),
            instrument_type=str(row[UDASH_COLUMNS["instrument_type"]]),
            timestamp=str(row[UDASH_COLUMNS["timestamp"]]),
            longitude_deg=float(row[UDASH_COLUMNS["longitude_deg"]]),
            latitude_deg=float(row[UDASH_COLUMNS["latitude_deg"]]),
        )


def summarize_profiles(
    df: pd.DataFrame,
    *,
    by: Optional[str] = "source_file",
) -> pd.DataFrame:
    """I summarise each profile so I know depth ranges and density extremes."""

    group_columns = [by] if by else [UDASH_COLUMNS["profile"], "source_file"]
    summary = (
        df.groupby(group_columns)
        .agg(
            n_levels=(UDASH_COLUMNS["pressure_dbar"], "size"),
            min_pressure=(UDASH_COLUMNS["pressure_dbar"], "min"),
            max_pressure=(UDASH_COLUMNS["pressure_dbar"], "max"),
            min_density=("Density_kg_m3", "min"),
            max_density=("Density_kg_m3", "max"),
        )
        .reset_index()
    )
    return summary


def plot_teos10_diagnostics(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = (
        "Absolute_Salinity_g_kg",
        "Conservative_Temp_degC",
        "Density_kg_m3",
    ),
    depth_column: str = UDASH_COLUMNS["depth_m"],
    share_depth_axis: bool = True,
    figsize: tuple[float, float] = (7.0, 9.0),
    marker: str = "o",
    marker_size: float = 0.3,
) -> tuple["matplotlib.figure.Figure", Sequence["matplotlib.axes.Axes"]]:
    """I draw scatter plots (tiny dots!) of each diagnostic against depth."""

    import matplotlib.pyplot as plt

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError("Missing required columns: " + ", ".join(missing_columns))

    if depth_column not in df.columns:
        raise KeyError(f"Depth column '{depth_column}' not found in DataFrame")

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=figsize,
        sharey=share_depth_axis,
    )

    if len(columns) == 1:
        axes = [axes]

    depth = df[depth_column]

    for ax, column in zip(axes, columns):
        ax.scatter(df[column], depth, s=marker_size, marker=marker)
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"{column} vs depth")
        ax.invert_yaxis()  # I like the surface at the top, deep water at the bottom.
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    return fig, axes


if __name__ == "__main__":
    # When I run "%run udash_density.py" this block executes the full pipeline.
    df = load_default_profile()

    file_name = PROFILE_TEXT_FILE.name
    print(f"Loaded {file_name} with {len(df)} levels")
    print()

    print("Profile overview:")
    print(preview_profile_metadata(df, max_rows=None))
    print()

    print("Sampled profile levels:")
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(preview_level_samples(df, samples=12))
    print()

    output_path = df.attrs.get("teos10_output_path")
    if output_path:
        print(f"Saved TEOS-10 variables to: {output_path}")
        print()

    print("Density summary:")
    print(
        summarize_profiles(df, by=None)[
            [
                UDASH_COLUMNS["profile"],
                "source_file",
                "n_levels",
                "min_pressure",
                "max_pressure",
                "min_density",
                "max_density",
            ]
        ]
    )

    import matplotlib.pyplot as plt

    fig, _ = plot_teos10_diagnostics(df)
    fig.suptitle(f"TEOS-10 diagnostics for {file_name}", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.show()

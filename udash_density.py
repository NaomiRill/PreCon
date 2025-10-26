"""Minimal helpers to compute seawater density from a single UDASH CTD file.

Drop this file next to your notebook and import :func:`load_udash_file`.

Example
-------
>>> from udash_density import load_udash_file
>>> df = load_udash_file("UDASH/NABOS_2015_AT_001.txt")
>>> df.head()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import gsw

DEFAULT_UDASH_FILE = Path("UDASH/ArcticOcean_phys_oce_1980.txt")

__all__ = [
    "UDASH_COLUMNS",
    "UDASHProfileMetadata",
    "DEFAULT_UDASH_FILE",
    "load_udash_file",
    "load_arctic_ocean_profile",
    "iter_profile_metadata",
    "summarize_profiles",
    "preview_profile_metadata",
    "plot_teos10_diagnostics",
]

# Column names used in UDASH tab-separated exports. Keeping them in one
# dictionary makes it easier to adapt if the dataset schema changes.
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
    """Basic metadata for a UDASH profile."""

    profile: int
    cruise: str
    station: str
    platform: str
    instrument_type: str
    timestamp: str
    longitude_deg: float
    latitude_deg: float


def _read_udash_text(path: Path, missing_value: float = -999.0) -> pd.DataFrame:
    """Read a UDASH text file into a DataFrame with missing values handled."""

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        na_values=[missing_value],
        engine="python",
    )
    return df


def _append_teos10_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Absolute Salinity, Conservative Temperature, and density."""

    SA = gsw.SA_from_SP(
        df[UDASH_COLUMNS["salinity_psu"]].to_numpy(),
        df[UDASH_COLUMNS["pressure_dbar"]].to_numpy(),
        df[UDASH_COLUMNS["longitude_deg"]].to_numpy(),
        df[UDASH_COLUMNS["latitude_deg"]].to_numpy(),
    )
    CT = gsw.CT_from_t(
        SA,
        df[UDASH_COLUMNS["temperature_degC"]].to_numpy(),
        df[UDASH_COLUMNS["pressure_dbar"]].to_numpy(),
    )
    rho = gsw.rho(SA, CT, df[UDASH_COLUMNS["pressure_dbar"]].to_numpy())

    df = df.copy()
    df["Absolute_Salinity_g_kg"] = SA
    df["Conservative_Temp_degC"] = CT
    df["Density_kg_m3"] = rho
    return df


def load_udash_file(
    path: str | Path,
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Load a single UDASH text file and compute density columns.

    Parameters
    ----------
    path:
        Path to a UDASH ``.txt`` file.
    missing_value:
        Numeric value representing missing data in the file. The default
        ``-999`` matches UDASH conventions.
    drop_na:
        If ``True`` (default), rows with missing pressure, temperature, or
        salinity are dropped before the TEOS-10 calculations.

    Returns
    -------
    pandas.DataFrame
        Table containing the raw UDASH variables plus Absolute Salinity,
        Conservative Temperature, and in-situ density (kg/m³).
    """

    path = Path(path)
    df = _read_udash_text(path, missing_value=missing_value)

    required = [
        UDASH_COLUMNS["salinity_psu"],
        UDASH_COLUMNS["temperature_degC"],
        UDASH_COLUMNS["pressure_dbar"],
        UDASH_COLUMNS["longitude_deg"],
        UDASH_COLUMNS["latitude_deg"],
    ]
    if drop_na:
        df = df.dropna(subset=required)

    df = _append_teos10_columns(df)
    df.insert(0, "source_file", path.name)
    return df


def load_arctic_ocean_profile(
    path: str | Path = DEFAULT_UDASH_FILE,
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Shortcut for the ``ArcticOcean_phys_oce_1980.txt`` UDASH profile."""

    return load_udash_file(
        path,
        missing_value=missing_value,
        drop_na=drop_na,
    )


def preview_profile_metadata(
    df: pd.DataFrame,
    *,
    max_rows: Optional[int] = 10,
) -> pd.DataFrame:
    """Return distinct profile metadata rows for quick inspection."""

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


def iter_profile_metadata(df: pd.DataFrame) -> Iterable[UDASHProfileMetadata]:
    """Iterate over unique profile metadata rows extracted from a DataFrame."""

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
    """Generate a quick summary for each profile or source file."""

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
) -> tuple["matplotlib.figure.Figure", Sequence["matplotlib.axes.Axes"]]:
    """Plot TEOS-10 diagnostic columns against depth using matplotlib."""

    import matplotlib.pyplot as plt

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if depth_column not in df.columns:
        raise KeyError(f"Depth column '{depth_column}' not found in DataFrame")

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=figsize,
        sharey=share_depth_axis,
        constrained_layout=True,
    )

    if len(columns) == 1:
        axes = [axes]

    depth = df[depth_column]

    for ax, column in zip(axes, columns):
        ax.plot(df[column], depth, marker=marker)
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"{column} vs depth")
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)

    return fig, axes


if __name__ == "__main__":
    df = load_arctic_ocean_profile()
    print("Loaded ArcticOcean_phys_oce_1980.txt with", len(df), "levels")
    print()
    print("Profile overview:")
    print(preview_profile_metadata(df, max_rows=None))
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
    fig.suptitle("TEOS-10 diagnostics for ArcticOcean_phys_oce_1980.txt", y=0.99)
    plt.show()

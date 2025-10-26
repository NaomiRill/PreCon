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
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
import gsw

DEFAULT_UDASH_FILE = Path("UDASH/ArcticOcean_phys_oce_1980.txt")

__all__ = [
    "UDASH_COLUMNS",
    "UDASHProfileMetadata",
    "DEFAULT_UDASH_FILE",
    "load_udash_file",
    "load_arctic_ocean_profile",
    "georeference_levels",
    "aggregate_profiles",
    "plot_profile_location_map",
    "plot_profile_variable_map",
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


def georeference_levels(
    df: pd.DataFrame,
    *,
    longitude_column: str = UDASH_COLUMNS["longitude_deg"],
    latitude_column: str = UDASH_COLUMNS["latitude_deg"],
    crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """Attach geographic point geometry to each UDASH level.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_udash_file`.
    longitude_column, latitude_column:
        Column names containing the geographic coordinates.
    crs:
        Coordinate reference system for the resulting GeoDataFrame. The
        default ``"EPSG:4326"`` corresponds to WGS84 longitude/latitude.

    Returns
    -------
    geopandas.GeoDataFrame
        Input data with an added ``geometry`` column of ``Point`` objects.
    """

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "geopandas and shapely are required for georeferencing; install them"
            " via `pip install geopandas shapely`."
        ) from exc

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(df[longitude_column].to_numpy(), df[latitude_column].to_numpy())
    ]

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    return gdf


def aggregate_profiles(
    df: pd.DataFrame,
    *,
    aggregator: str | Callable[[pd.Series], float] = "mean",
    value_columns: Sequence[str] = (
        "Density_kg_m3",
        "Conservative_Temp_degC",
        "Absolute_Salinity_g_kg",
    ),
) -> pd.DataFrame:
    """Collapse repeated profile levels to unique geographic points.

    The function keeps one row per ``(profile, source_file)`` pair, retaining
    the first occurrence of metadata/coordinate columns and aggregating the
    TEOS-10 diagnostic columns with ``aggregator``.
    """

    group_keys = [UDASH_COLUMNS["profile"], "source_file"]

    aggregations: dict[str, Callable | str] = {
        UDASH_COLUMNS["longitude_deg"]: "first",
        UDASH_COLUMNS["latitude_deg"]: "first",
        UDASH_COLUMNS["timestamp"]: "first",
        UDASH_COLUMNS["cruise"]: "first",
        UDASH_COLUMNS["station"]: "first",
        UDASH_COLUMNS["platform"]: "first",
        UDASH_COLUMNS["instrument_type"]: "first",
    }
    for column in value_columns:
        if column in df.columns:
            aggregations[column] = aggregator

    aggregated = df.groupby(group_keys, as_index=False).agg(aggregations)
    return aggregated


def _prepare_map_axes(ax=None):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax

    fig, ax = plt.subplots(figsize=(6, 6))
    return ax


def plot_profile_location_map(
    df: pd.DataFrame,
    *,
    ax=None,
    annotate: bool = False,
    **scatter_kwargs,
):
    """Plot the geographic location of each profile on a latitude/longitude map."""

    ax = _prepare_map_axes(ax)

    aggregated = aggregate_profiles(df)
    lon = aggregated[UDASH_COLUMNS["longitude_deg"]]
    lat = aggregated[UDASH_COLUMNS["latitude_deg"]]
    scatter_defaults = {"s": 120, "color": "tab:blue", "edgecolor": "black"}
    scatter_defaults.update(scatter_kwargs)

    ax.scatter(lon, lat, **scatter_defaults)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("UDASH profile locations")

    if annotate:
        for _, row in aggregated.iterrows():
            label = f"{row[UDASH_COLUMNS['cruise']]} / {row[UDASH_COLUMNS['station']]}"
            ax.annotate(label, (row[UDASH_COLUMNS["longitude_deg"]], row[UDASH_COLUMNS["latitude_deg"]]))

    return ax


def plot_profile_variable_map(
    df: pd.DataFrame,
    *,
    column: str = "Density_kg_m3",
    aggregator: str | Callable[[pd.Series], float] = "mean",
    ax=None,
    cmap: str = "viridis",
    colorbar: bool = True,
    **scatter_kwargs,
):
    """Create a map coloured by an aggregated TEOS-10 diagnostic column."""

    import matplotlib.pyplot as plt

    ax = _prepare_map_axes(ax)

    aggregated = aggregate_profiles(df, aggregator=aggregator, value_columns=(column,))
    if column not in aggregated:
        raise KeyError(f"Column '{column}' not found in DataFrame; available columns: {list(df.columns)}")

    lon = aggregated[UDASH_COLUMNS["longitude_deg"]]
    lat = aggregated[UDASH_COLUMNS["latitude_deg"]]
    values = aggregated[column]

    scatter_defaults = {"s": 150, "cmap": cmap, "edgecolor": "black"}
    scatter_defaults.update(scatter_kwargs)

    scatter = ax.scatter(lon, lat, c=values, **scatter_defaults)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"UDASH profile {column} ({aggregator})")

    if colorbar:
        plt.colorbar(scatter, ax=ax, label=column)

    return ax


if __name__ == "__main__":
    df = load_arctic_ocean_profile()
    print("Loaded ArcticOcean_phys_oce_1980.txt with", len(df), "levels")
    print()
    print(df.head())
    print()
    print("Density summary:")
    print(
        summarize_profiles(df)[
            [
                "n_levels",
                "min_pressure",
                "max_pressure",
                "min_density",
                "max_density",
            ]
        ]
    )

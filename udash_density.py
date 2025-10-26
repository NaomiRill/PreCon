"""Helpers to load one UDASH profile, compute TEOS-10 values, and plot them.

Every block in this file is intentionally annotated with comments so the code can
serve as a learning resource while also being easy to adapt for other UDASH
profiles (for example the yearly Arctic Ocean releases from 1980-2015)."""

# The following future import delays evaluation of type annotations until run
# time. This lets us use the modern ``list[str]`` / ``tuple[float, ...]`` style
# annotations even when the module is imported on Python 3.10, which would
# otherwise require importing ``from __future__ import annotations`` manually.
from __future__ import annotations

# ``dataclasses`` provides the lightweight container we use for metadata.
from dataclasses import dataclass
# ``pathlib`` gives us high-level filesystem helpers for locating UDASH files.
from pathlib import Path
# Typing helpers keep public function signatures clear and self-documenting.
from typing import Iterable, Optional, Sequence

# ``warnings`` is used to inform the user whenever the data requires cleaning
# (for example when negative salinity values are encountered).
import warnings

# ``numpy`` powers the numerical conversions; ``pandas`` handles tabular data;
# and ``gsw`` implements the TEOS-10 algorithms required for seawater analysis.
import numpy as np
import pandas as pd
import gsw

# Default location of the UDASH text file that started this project. All file
# operations are relative to this constant so changing years is as easy as
# pointing it at a different ``*.txt`` file.
DEFAULT_UDASH_FILE = Path("UDASH/ArcticOcean_phys_oce_1980.txt")

# ``__all__`` lists the public names re-exported when the module is imported
# via ``from udash_density import *``. This also doubles as a quick reference of
# what the notebook author is expected to use.
__all__ = [
    "UDASH_COLUMNS",
    "UDASHProfileMetadata",
    "DEFAULT_UDASH_FILE",
    "load_udash_file",
    "load_arctic_ocean_profile",
    "iter_profile_metadata",
    "summarize_profiles",
    "preview_profile_metadata",
    "preview_level_samples",
    "plot_teos10_diagnostics",
    "save_teos10_variables",
]

UDASH_COLUMNS = {
    # Unique identifiers describing a single cast in the UDASH dataset.
    "profile": "Prof_no",
    "cruise": "Cruise",
    "station": "Station",
    "platform": "Platform",
    "instrument_type": "Type",
    # The timestamp column often encodes both date and time. We reference it by
    # name so that switching datasets stays painless.
    "timestamp": "yyyy-mm-ddThh:mm",
    # Geographic coordinates in decimal degrees.
    "longitude_deg": "Longitude_[deg]",
    "latitude_deg": "Latitude_[deg]",
    # Pressure (in decibars) and depth (in metres) describe the sampling levels.
    "pressure_dbar": "Pressure_[dbar]",
    "depth_m": "Depth_[m]",
    # In-situ temperature and practical salinity are the inputs to TEOS-10.
    "temperature_degC": "Temp_[°C]",
    "salinity_psu": "Salinity_[psu]",
    # Quality flags can be helpful when filtering the data before processing.
    "quality_flag": "QF",
}


@dataclass
class UDASHProfileMetadata:
    """Container describing one UDASH profile's identifying information."""

    # The dataclass fields mirror the keys defined above so each metadata row
    # can be stored and iterated over with clear attribute access in notebooks.
    profile: int
    cruise: str
    station: str
    platform: str
    instrument_type: str
    timestamp: str
    longitude_deg: float
    latitude_deg: float


def _infer_profile_year(df: pd.DataFrame) -> str:
    """Infer the profile year from the timestamp column if possible.

    This helper isolates the logic for extracting a year so the same behaviour
    is reused by any future export routines. The function intentionally returns
    ``"unknown"`` rather than raising when the year cannot be derived because
    some older UDASH exports omit timestamps entirely.
    """

    # Look up the column name once for clarity and to avoid repeated dictionary
    # indexing in the rest of the function.
    timestamp_column = UDASH_COLUMNS["timestamp"]
    if timestamp_column not in df.columns:
        # If the column is missing we cannot determine the year.
        return "unknown"

    # Convert all timestamps to strings, dropping NaNs along the way. This
    # allows the code to work with mixed string/datetime columns seamlessly.
    timestamps = df[timestamp_column].dropna().astype(str)
    if timestamps.empty:
        return "unknown"

    first_value = timestamps.iloc[0]
    try:
        # Use pandas to parse the string into a Timestamp and extract the year.
        year = pd.to_datetime(first_value).year
    except (TypeError, ValueError):
        # Some UDASH strings look like ``1980-01-29T13:00``; slicing the first
        # four characters still yields the year when parsing fails.
        year = str(first_value)[:4]
    return str(year)


def save_teos10_variables(
    df: pd.DataFrame,
    *,
    source_path: Path,
    output_root: Path = Path("Density"),
) -> Path:
    """Persist the computed TEOS-10 variables to a year-specific text file.

    The export includes longitude, latitude, pressure, depth, and the three
    TEOS-10 diagnostics produced by :func:`load_udash_file`. Saving the data is
    optional but recommended so downstream geospatial steps have a tidy text
    file to ingest.
    """

    # Determine the folder for the current profile using the inferred year. The
    # directory is created on demand and re-used across runs.
    year = _infer_profile_year(df)
    target_dir = output_root / f"{year}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Compose the final file name from the original UDASH file name. For
    # example, ``ArcticOcean_phys_oce_1980.txt`` becomes
    # ``Density/1980/ArcticOcean_phys_oce_1980_teos10.txt``.
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
    # ``available_columns`` guards against situations where a UDASH file omits
    # certain variables. Only the intersection of available columns is written
    # to disk so the export step never fails unexpectedly.
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
        When ``save_output`` is ``True`` the returned DataFrame contains a
        ``"teos10_output_path"`` entry in ``df.attrs`` pointing at the saved
        text file.
    """

    # Convert the provided path into a ``Path`` object so we get convenient
    # filesystem behaviour (``.stem``, ``.name``) regardless of the input type.
    path = Path(path)
    # ``read_csv`` handles whitespace-delimited files as long as we pass
    # ``delim_whitespace=True``. Any ``-999`` entries (the UDASH sentinel for
    # missing data) are replaced with NaN automatically via ``na_values``.
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        na_values=[missing_value],
        engine="python",
    )

    salinity_column = UDASH_COLUMNS["salinity_psu"]
    if salinity_column in df.columns:
        # ``to_numeric`` converts the column to floats while coercing any
        # unexpected strings to NaN. This keeps TEOS-10 from crashing later on.
        salinity = pd.to_numeric(df[salinity_column], errors="coerce")
        negative_mask = salinity < 0
        if negative_mask.any():
            # Negative practical salinities are not physical. We flag them once
            # per file and then mask them so they are ignored by the TEOS-10
            # functions and by the subsequent NaN filtering.
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
        # Removing rows with missing inputs prevents ``gsw`` from receiving NaN
        # values, which would otherwise propagate through the calculations.
        df = df.dropna(subset=required)

    # Convert the cleaned columns into numpy arrays so we can pass them to the
    # TEOS-10 routines. ``dtype=float`` ensures consistent precision.
    salinity = df[salinity_column].to_numpy(dtype=float)
    pressure = df[UDASH_COLUMNS["pressure_dbar"]].to_numpy(dtype=float)
    longitude = df[UDASH_COLUMNS["longitude_deg"]].to_numpy(dtype=float)
    latitude = df[UDASH_COLUMNS["latitude_deg"]].to_numpy(dtype=float)
    temperature = df[UDASH_COLUMNS["temperature_degC"]].to_numpy(dtype=float)

    # ``gsw.SA_from_SP`` converts practical salinity (unitless) to Absolute
    # Salinity (grams per kilogram) using pressure and geographic context.
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    # ``gsw.CT_from_t`` converts in-situ temperature to Conservative Temperature
    # (also in degrees Celsius).
    CT = gsw.CT_from_t(SA, temperature, pressure)
    # ``gsw.rho`` calculates in-situ density (kg/m³) from Absolute Salinity,
    # Conservative Temperature, and pressure.
    rho = gsw.rho(SA, CT, pressure)

    # Attach the TEOS-10 output as new columns so the original UDASH variables
    # remain available alongside the computed diagnostics.
    df = df.copy()
    df["Absolute_Salinity_g_kg"] = SA
    df["Conservative_Temp_degC"] = CT
    df["Density_kg_m3"] = rho
    # Insert the source file name as the first column for quick reference in
    # notebooks that display multiple casts at once.
    df.insert(0, "source_file", path.name)

    if save_output:
        # Save the calculated values to ``Density/<year>/...`` for geospatial
        # workflows. The resulting path is stored in ``df.attrs`` so notebooks
        # can display it back to the reader.
        output_path = save_teos10_variables(
            df,
            source_path=path,
            output_root=output_root,
        )
        df.attrs["teos10_output_path"] = str(output_path)
    return df


def load_arctic_ocean_profile(
    path: str | Path = DEFAULT_UDASH_FILE,
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
    save_output: bool = True,
    output_root: Path = Path("Density"),
) -> pd.DataFrame:
    """Shortcut for the ``ArcticOcean_phys_oce_1980.txt`` UDASH profile.

    Updating ``DEFAULT_UDASH_FILE`` (for example to
    ``Path("UDASH/ArcticOcean_phys_oce_1991.txt")``) automatically makes this
    helper load the new year without touching the function body.
    """

    return load_udash_file(
        path,
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
    """Return distinct profile metadata rows for quick inspection.

    Displaying metadata (source file, cruise, station, platform, instrument)
    alongside the TEOS-10 diagnostics helps spot if multiple casts are present
    in one file.
    """

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
    """Return evenly spaced profile rows for console previews.

    The goal is to see the full column structure without printing thousands of
    rows. Twelve samples usually give a good sense of the dataset.
    """

    if samples <= 0 or df.empty:
        # Nothing to sample; return an empty frame with the correct columns.
        return df.head(0)

    if samples >= len(df):
        # If the caller requests more samples than available rows, fall back to
        # returning the entire DataFrame.
        return df

    # ``linspace`` spreads indices evenly across the DataFrame; rounding them
    # back to integers gives a representative subset of levels.
    indices = np.linspace(0, len(df) - 1, samples)
    unique_indices = np.unique(indices.round().astype(int))
    return df.iloc[unique_indices]


def iter_profile_metadata(df: pd.DataFrame) -> Iterable[UDASHProfileMetadata]:
    """Iterate over unique profile metadata rows extracted from a DataFrame.

    This generator yields each unique profile exactly once. It is particularly
    convenient when preparing map layers or summary tables.
    """

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
    """Generate a quick summary for each profile or source file.

    The summary currently reports the number of levels, pressure range, and
    density range. Additional metrics (minimum/maximum temperature, etc.) can
    be added by extending the ``agg`` call below.
    """

    group_columns = [by] if by else [UDASH_COLUMNS["profile"], "source_file"]
    summary = (
        df.groupby(group_columns)
        .agg(
            # Count how many depth levels are present in each profile.
            n_levels=(UDASH_COLUMNS["pressure_dbar"], "size"),
            # Report the shallowest and deepest pressure levels observed.
            min_pressure=(UDASH_COLUMNS["pressure_dbar"], "min"),
            max_pressure=(UDASH_COLUMNS["pressure_dbar"], "max"),
            # Density range is often used to check for outliers.
            min_density=("Density_kg_m3", "min"),
            max_density=("Density_kg_m3", "max"),
        )
        # Restore the grouping columns as regular columns for easy printing.
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
    marker_size: float = 0.6,
) -> tuple["matplotlib.figure.Figure", Sequence["matplotlib.axes.Axes"]]:
    """Plot TEOS-10 diagnostic columns against depth using matplotlib.

    The scatter plots emphasise individual sampling levels. Adjust ``marker`` or
    ``marker_size`` if you prefer different aesthetics.
    """

    import matplotlib.pyplot as plt

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if depth_column not in df.columns:
        raise KeyError(f"Depth column '{depth_column}' not found in DataFrame")

    # Create one subplot per diagnostic. ``sharey`` keeps the depth axis
    # consistent which makes visual comparisons easier.
    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=figsize,
        sharey=share_depth_axis,
    )

    if len(columns) == 1:
        axes = [axes]

    # Grab the depth column once so the loop below can reuse it without
    # repeating dictionary lookups.
    depth = df[depth_column]

    for ax, column in zip(axes, columns):
        # ``scatter`` draws one point per level without connecting lines.
        ax.scatter(df[column], depth, s=marker_size, marker=marker)
        ax.set_xlabel(column.replace("_", " "))
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"{column} vs depth")
        # Depth increases with depth, so invert the axis to show the surface on
        # top and the deepest levels at the bottom.
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.5)

    # ``tight_layout`` prevents overlapping labels; the manual adjustment keeps
    # space available for the shared figure title added later.
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    return fig, axes


if __name__ == "__main__":
    # Execute the full workflow when the module is run as a script (for
    # example via ``%run udash_density.py`` inside Jupyter). This section keeps
    # the notebook user experience reproducible while doubling as executable
    # documentation of the helper functions.
    df = load_arctic_ocean_profile()
    print("Loaded ArcticOcean_phys_oce_1980.txt with", len(df), "levels")
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
    fig.suptitle("TEOS-10 diagnostics for ArcticOcean_phys_oce_1980.txt", y=0.98)
    # ``tight_layout`` with ``rect`` leaves room for the figure title. The
    # ``plt.show()`` call finally renders the plots when running as a script.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.show()

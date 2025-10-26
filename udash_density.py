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

    timestamp_column = UDASH_COLUMNS["timestamp"]  # I reference the canonical timestamp column name.
    if timestamp_column not in df.columns:  # I check whether the expected timestamp column even exists.
        return "unknown"  # I bail out immediately if the timestamp column never existed.

    timestamps = df[timestamp_column].dropna().astype(str)  # I drop missing timestamps and coerce everything to strings.
    if timestamps.empty:  # I make sure there is at least one timestamp left to inspect.
        return "unknown"  # I cannot infer anything when there are zero usable timestamps.

    first_value = timestamps.iloc[0]  # I just need one timestamp to infer the year.
    try:  # I attempt the clean timestamp parsing route first.
        year = pd.to_datetime(first_value).year  # I lean on pandas to parse the timestamp for me.
    except (TypeError, ValueError):  # I recover gracefully when pandas cannot parse the timestamp.
        year = str(first_value)[:4]  # I fall back to string slicing if parsing explodes.
    return str(year)  # I always give back a string so the directory naming stays predictable.


def save_teos10_variables(
    df: pd.DataFrame,
    *,
    source_path: Path,
    output_root: Path = Path("Density"),
) -> Path:
    """I write the TEOS-10 results to Density/<year>/... so mapping is easy later."""

    year = _infer_profile_year(df)  # I figure out where the file should live inside Density/.
    target_dir = output_root / f"{year}"  # I compose the directory path for that year.
    target_dir.mkdir(parents=True, exist_ok=True)  # I ensure the directory exists before writing.

    output_path = target_dir / f"{source_path.stem}_teos10.txt"

    export_columns = [
        UDASH_COLUMNS["longitude_deg"],  # I always keep longitude because mapping depends on it.
        UDASH_COLUMNS["latitude_deg"],  # I keep latitude for the same mapping reason.
        UDASH_COLUMNS["pressure_dbar"],  # I save pressure so I can rebuild the water column later.
        UDASH_COLUMNS["depth_m"],  # I keep the depth column as a second vertical reference.
        "Absolute_Salinity_g_kg",  # I export the TEOS-10 absolute salinity results.
        "Conservative_Temp_degC",  # I export conservative temperature as well.
        "Density_kg_m3",  # I export density because that is the main product.
    ]
    available_columns = [column for column in export_columns if column in df.columns]  # I filter out anything missing just in case.

    df[available_columns].to_csv(
        output_path,
        sep="\t",
        index=False,
        float_format="%.6f",
    )  # I actually write the tab-separated file with fixed floating precision.
    return output_path  # I return the path so the caller knows where the file landed.


def load_udash_file(
    path: str | Path,
    *,
    missing_value: float = -999.0,
    drop_na: bool = True,
    save_output: bool = True,
    output_root: Path = Path("Density"),
) -> pd.DataFrame:
    """I read the UDASH text file, run TEOS-10, and optionally save the results."""

    path = Path(path)  # I convert the incoming path right away so I can use pathlib conveniences.
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        na_values=[missing_value],
        engine="python",
    )  # I read the UDASH text file using pandas with whitespace separation and the UDASH missing value.

    salinity_column = UDASH_COLUMNS["salinity_psu"]  # I keep the salinity column name handy.
    if salinity_column in df.columns:  # I only run the cleaning logic when the column is present.
        salinity = pd.to_numeric(df[salinity_column], errors="coerce")  # I coerce non-numeric salinity values to NaN.
        negative_mask = salinity < 0  # I find any negative salinity entries that should not exist.
        if negative_mask.any():  # I only warn when I truly spotted negative numbers.
            warnings.warn(
                "Negative practical salinity values found; treating them as missing.",
                RuntimeWarning,
                stacklevel=2,
            )  # I shout a warning so future me knows the file looked suspicious.
            salinity = salinity.mask(negative_mask)  # I replace negative values with NaN so TEOS-10 never sees them.
        df[salinity_column] = salinity  # I stash the cleaned salinity back into the DataFrame.

    required = [
        salinity_column,
        UDASH_COLUMNS["temperature_degC"],
        UDASH_COLUMNS["pressure_dbar"],
        UDASH_COLUMNS["longitude_deg"],
        UDASH_COLUMNS["latitude_deg"],
    ]  # I list every column TEOS-10 will require later on.
    if drop_na:  # I optionally drop rows based on the caller's preference.
        df = df.dropna(subset=required)  # I drop rows missing any of the critical TEOS-10 inputs.

    salinity = df[salinity_column].to_numpy(dtype=float)  # I prepare NumPy arrays for gsw.
    pressure = df[UDASH_COLUMNS["pressure_dbar"]].to_numpy(dtype=float)  # I extract pressure values as floats.
    longitude = df[UDASH_COLUMNS["longitude_deg"]].to_numpy(dtype=float)  # I pull out the longitude values.
    latitude = df[UDASH_COLUMNS["latitude_deg"]].to_numpy(dtype=float)  # I grab the latitude values.
    temperature = df[UDASH_COLUMNS["temperature_degC"]].to_numpy(dtype=float)  # I extract in-situ temperature.

    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)  # I compute Absolute Salinity from the practical salinity and coordinates.
    CT = gsw.CT_from_t(SA, temperature, pressure)  # I convert Absolute Salinity and in-situ temperature to Conservative Temperature.
    rho = gsw.rho(SA, CT, pressure)  # I derive in-situ density using the TEOS-10 equation of state.

    df = df.copy()  # I copy so I do not mutate the original DataFrame reference.
    df["Absolute_Salinity_g_kg"] = SA  # I append the Absolute Salinity results.
    df["Conservative_Temp_degC"] = CT  # I append the Conservative Temperature results.
    df["Density_kg_m3"] = rho  # I append the density results.
    df.insert(0, "source_file", path.name)  # I record the source filename as the first column for provenance.

    if save_output:  # I only persist to disk when the caller asked for it.
        output_path = save_teos10_variables(
            df,
            source_path=path,
            output_root=output_root,
        )  # I push the TEOS-10 variables to disk right after computing them.
        df.attrs["teos10_output_path"] = str(output_path)  # I stash the save location in attrs so notebooks can report it.
    return df  # I hand back the enriched DataFrame for further analysis.


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
    )  # I simply forward everything to load_udash_file so there is one code path.


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
    ]  # I decide up front which columns give me the quick overview I want.
    preview = df.loc[:, columns].drop_duplicates().reset_index(drop=True)  # I drop duplicates so repeated levels do not spam.
    if max_rows is not None:  # I trim the preview when a maximum was provided.
        preview = preview.head(max_rows)  # I optionally clip the preview for readability.
    return preview  # I return the preview DataFrame so the notebook can display it.


def preview_level_samples(
    df: pd.DataFrame,
    *,
    samples: int = 12,
) -> pd.DataFrame:
    """Instead of printing everything I just peek at evenly spaced rows."""

    if samples <= 0 or df.empty:  # I check the edge cases where sampling would return nothing meaningful.
        return df.head(0)  # I return an empty slice when sampling makes no sense.

    if samples >= len(df):  # I avoid extra work when sampling would include every row.
        return df  # I give back the whole DataFrame when sampling would include every row anyway.

    indices = np.linspace(0, len(df) - 1, samples)  # I generate evenly spaced float indices across the DataFrame.
    unique_indices = np.unique(indices.round().astype(int))  # I round to integers and ensure uniqueness for indexing.
    return df.iloc[unique_indices]  # I pick those rows to represent the full profile without flooding the console.


def iter_profile_metadata(df: pd.DataFrame) -> Iterable[UDASHProfileMetadata]:
    """I yield one UDASHProfileMetadata per unique profile in the DataFrame."""

    seen = set()  # I keep track of which profile/source pairs I already emitted.
    for _, row in df.iterrows():  # I walk through every row to find the first instance of each profile.
        key = (row[UDASH_COLUMNS["profile"]], row["source_file"])  # I identify each unique profile by its number and file.
        if key in seen:  # I skip profiles I have already produced.
            continue  # I skip duplicates so each profile appears exactly once.
        seen.add(key)  # I record that this profile has now been yielded.
        yield UDASHProfileMetadata(
            profile=int(row[UDASH_COLUMNS["profile"]]),
            cruise=str(row[UDASH_COLUMNS["cruise"]]),
            station=str(row[UDASH_COLUMNS["station"]]),
            platform=str(row[UDASH_COLUMNS["platform"]]),
            instrument_type=str(row[UDASH_COLUMNS["instrument_type"]]),
            timestamp=str(row[UDASH_COLUMNS["timestamp"]]),
            longitude_deg=float(row[UDASH_COLUMNS["longitude_deg"]]),
            latitude_deg=float(row[UDASH_COLUMNS["latitude_deg"]]),
        )  # I emit a dataclass instance so downstream code can use attributes instead of raw dicts.


def summarize_profiles(
    df: pd.DataFrame,
    *,
    by: Optional[str] = "source_file",
) -> pd.DataFrame:
    """I summarise each profile so I know depth ranges and density extremes."""

    group_columns = [by] if by else [UDASH_COLUMNS["profile"], "source_file"]  # I decide how to group the summary rows.
    summary = (
        df.groupby(group_columns)
        .agg(
            n_levels=(UDASH_COLUMNS["pressure_dbar"], "size"),  # I count how many depth levels exist.
            min_pressure=(UDASH_COLUMNS["pressure_dbar"], "min"),  # I find the shallowest pressure.
            max_pressure=(UDASH_COLUMNS["pressure_dbar"], "max"),  # I find the deepest pressure.
            min_density=("Density_kg_m3", "min"),  # I capture the minimum density.
            max_density=("Density_kg_m3", "max"),  # I capture the maximum density.
        )
        .reset_index()
    )  # I assemble a small DataFrame that summarises the key stats.
    return summary  # I return the summary so notebooks can display or export it.


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

    missing_columns = [column for column in columns if column not in df.columns]  # I check that every requested column exists.
    if missing_columns:  # I guard against plotting without the necessary inputs.
        raise KeyError("Missing required columns: " + ", ".join(missing_columns))  # I fail early if something is missing.

    if depth_column not in df.columns:  # I double-check that the vertical axis exists.
        raise KeyError(f"Depth column '{depth_column}' not found in DataFrame")  # I refuse to plot without depth information.

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=figsize,
        sharey=share_depth_axis,
    )  # I create one subplot per diagnostic with optional shared depth axis.

    if len(columns) == 1:  # I normalise the axes object when there is only one subplot.
        axes = [axes]  # I normalise axes to a list so later loops do not need special cases.

    depth = df[depth_column]  # I reuse the depth series for every subplot.

    for ax, column in zip(axes, columns):  # I iterate over each subplot/column pair to render the scatter plots.
        ax.scatter(df[column], depth, s=marker_size, marker=marker)  # I plot tiny dots for each depth level.
        ax.set_xlabel(column.replace("_", " "))  # I swap underscores for spaces to make labels readable.
        ax.set_ylabel("Depth [m]")  # I label the shared depth axis.
        ax.set_title(f"{column} vs depth")  # I give each subplot its own title.
        ax.invert_yaxis()  # I flip the axis so shallow water sits at the top.
        ax.grid(True, linestyle="--", alpha=0.5)  # I add a faint grid to aid interpretation.

    fig.tight_layout()  # I squeeze the layout so labels do not overlap.
    fig.subplots_adjust(top=0.92)  # I leave room for a shared title if I want one later.

    return fig, axes  # I return both figure and axes so the caller can tweak them.


if __name__ == "__main__":  # I only run this orchestration block during direct execution.
    # I execute this block when I run "%run udash_density.py" in a notebook.
    df = load_default_profile()  # I load the configured profile and compute TEOS-10 diagnostics.

    file_name = PROFILE_TEXT_FILE.name  # I keep the filename handy for printouts and titles.
    print(f"Loaded {file_name} with {len(df)} levels")  # I confirm how many depth levels were found.
    print()

    print("Profile overview:")  # I announce the metadata preview.
    print(preview_profile_metadata(df, max_rows=None))  # I show every distinct profile metadata row.
    print()

    print("Sampled profile levels:")  # I announce the level sampling section.
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(preview_level_samples(df, samples=12))  # I print a spread of representative depth levels with all columns.
    print()

    output_path = df.attrs.get("teos10_output_path")  # I retrieve where the TEOS-10 export was saved.
    if output_path:  # I only announce the save path when the export actually happened.
        print(f"Saved TEOS-10 variables to: {output_path}")  # I remind myself of the export location.
        print()

    print("Density summary:")  # I announce the summary statistics table.
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
        ]  # I narrow the columns to the ones I care about in the console.
    )

    import matplotlib.pyplot as plt  # I pull in matplotlib only when I actually plot.

    fig, _ = plot_teos10_diagnostics(df)  # I create the scatter plots for the TEOS-10 diagnostics.
    fig.suptitle(f"TEOS-10 diagnostics for {file_name}", y=0.98)  # I add a shared title that mentions the file.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])  # I tighten again so the title and plots fit nicely.
    plt.show()  # I display the figure in the notebook output.

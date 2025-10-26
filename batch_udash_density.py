"""Little script that lets me run udash_density on several UDASH text files."""

from __future__ import annotations  # I keep postponed annotations for consistency with the helper module.

from pathlib import Path  # I prefer Path objects for manipulating the file list.
from typing import Iterable, Sequence  # Typing notes keep the loops readable.

import pandas as pd  # I rely on pandas to concatenate the individual profile tables.

import udash_density  # I reuse every TEOS-10 routine from the main helper.

# I list the UDASH profile files I want to process in this batch run.
# Later on I can simply append additional years to this tuple and rerun the script.
PROFILE_FILES: Sequence[Path] = (
    udash_density.PROFILE_TEXT_FILE,  # I default to the same file as the main helper for now.
)


def _ensure_sequence(paths: Sequence[Path | str]) -> Iterable[Path]:
    """I normalise whatever iterable I get into Path objects."""

    for entry in paths:  # I iterate over each entry that was configured.
        yield Path(entry)  # I coerce to Path so downstream code never juggles raw strings.


def process_profiles(paths: Sequence[Path | str] = PROFILE_FILES) -> pd.DataFrame:
    """I loop through every configured profile and stack the processed tables."""

    processed_tables = []  # I accumulate each processed DataFrame here.
    for path in _ensure_sequence(paths):  # I iterate through the cleaned-up list of paths.
        df = udash_density.load_udash_file(path)  # I compute the TEOS-10 diagnostics for this profile.
        processed_tables.append(df)  # I remember the table so I can concatenate everything at the end.
    if not processed_tables:  # I guard against the edge case of an empty profile list.
        return pd.DataFrame()  # I return an empty DataFrame when there was nothing to process.
    return pd.concat(processed_tables, ignore_index=True)  # I merge the processed tables into a single DataFrame.


if __name__ == "__main__":  # I only run the batch workflow when the script is executed directly.
    combined = process_profiles()  # I kick off the batch processing using the default profile list.
    if combined.empty:  # I let myself know when no data came back.
        print("No profiles were processed. Check PROFILE_FILES.")
    else:
        print(f"Processed {len(combined)} depth levels from {combined['source_file'].nunique()} profile file(s).")
        print()
        print("Summary by source file:")
        print(
            udash_density.summarize_profiles(combined, by="source_file")[
                [
                    "source_file",
                    "n_levels",
                    "min_pressure",
                    "max_pressure",
                    "min_density",
                    "max_density",
                ]
            ]
        )

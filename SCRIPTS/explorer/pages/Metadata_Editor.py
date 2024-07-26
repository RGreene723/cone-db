from pathlib import Path
from datetime import datetime

import pandas as pd
import json

import streamlit as st

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Metadata Editor", page_icon="📊", layout="wide")

st.title("Bulk Metadata Editor")

metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

# col1, col2, col3 = st.columns([0.2, 0.1, 0.7], vertical_alignment="top", gap="small")


@st.cache_data(show_spinner=True)
def load_metadata():

    all_metadata = []
    for metadata_path in metadata_path_map.values():
        all_metadata.append(json.load(open(metadata_path)))

    df = pd.DataFrame(all_metadata, index=list(metadata_path_map.keys())).sort_values(
        by=["date"]
    )

    df["** DELETE FILE"] = False
    df["material_id"] = None

    return df


def save_metadata():
    bar = st.progress(0, "Loading metadata ...")

    bar.progress(0, "Saving metadata ...")
    # Go through the dataframe row by row & save each file
    files_saved = 0

    # Remove the ** DELETE FILE column
    df = df.drop(columns=["** DELETE FILE"])

    for index, row in df.iterrows():
        # Get the path to the metadata file
        path = metadata_path_map[str(index)]
        # Convert the dataframe row to a dictionary
        row = row.to_dict()
        # Save the dictionary as JSON
        with open(path, "w") as f:
            json.dump(row, f, indent=4)
        files_saved += 1
        bar.progress(
            files_saved / len(metadata_path_map),
            f"Saving metadata for {path.stem}",
        )
    bar.progress(1.0, "Metadata saved")


st.sidebar.markdown("#### Save metadata")
st.sidebar.button("Save", on_click=save_metadata, use_container_width=True)
st.sidebar.button("Reload", on_click=st.cache_data.clear, use_container_width=True)
st.divider()

df = load_metadata()


df = st.data_editor(
    df,
    use_container_width=True,
    height=650,
    column_order=[
        "** DELETE FILE",
        "date",
        "material_id",
        "comments",
        "material_name",
        "specimen_description",
        "specimen_prep",
        "specimen_number",
        "report_name",
        "heat_flux_kW/m2",
        "laboratory",
        "operator",
        "test_start_time_s",
        "test_end_time_s",
        "c_factor",
    ],
)

st.sidebar.markdown("#### Delete files")
st.sidebar.markdown(
    "Select files by clicking the checkbox next to the file name, then click **Delete files** to delete the selected files."
)

# file_to_delete = st.sidebar.multiselect(
#     "File(s) to delete", options=list(metadata_path_map.keys())
# )


def delete_files():
    files_to_delete = df[df["** DELETE FILE"]].index
    for file in files_to_delete:
        metadata_path_map[file].unlink()
        metadata_path_map[file].with_suffix(".csv").unlink()
    st.cache_data.clear()
    st.success(f"{len(files_to_delete)} files deleted")


st.sidebar.button("Delete files", on_click=delete_files, use_container_width=True)


def export_metadata():
    bar = st.progress(0, "Exporting metadata ...")
    files_exported = 0
    for index, row in df.iterrows():
        row = row.to_dict()
        path = metadata_path_map[str(index)]

        # parse iso format datetime and just keep the date (no time)
        d = datetime.strptime(row["date"], "%Y-%m-%dT%H:%M:%S")
        year = d.strftime("%Y")
        date = d.strftime("%Y-%m-%d")

        if row.get("material_id") is None:
            continue

        if row.get("specimen_number") is not None:
            new_filename = f"{date}-{row['material_id']}-r{row['specimen_number']}.json"
        else:
            new_filename = f"{date}-{row['material_id']}.json"

        # include the old filename in the metadata
        row["prev_filename"] = path.name

        with open(path.parent / year / new_filename, "w") as f:
            json.dump(row, f, indent=4)

        files_exported += 1
        bar.progress(
            files_exported / len(metadata_path_map),
            f"Exporting metadata for {path.stem}",
        )
    bar.progress(1.0, f"Metadata exported ({files_exported} files)")


st.sidebar.markdown("#### Export metadata")
st.sidebar.button("Export", on_click=export_metadata, use_container_width=True)

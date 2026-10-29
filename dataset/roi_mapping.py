import pandas as pd


def get_roi_names(name_path):
    """Load ROI names from TauRegionList.csv and return canonical SUVR column names."""
    names_df = pd.read_csv(name_path)
    attrs = []
    for name in list(names_df.FS_label):
        name = name.replace("-", "_")
        attrs.append(name.upper() + "_SUVR")
    return attrs


def map_region_columns(df, region_names):
    """Map FS_label ROI names to concrete CSV columns."""
    cols = []
    for region in region_names:
        region_clean = region.replace("-", "_").upper()
        patterns = [
            f"CTX_LH_{region_clean}_SUVR",
            f"CTX_RH_{region_clean}_SUVR",
            f"LEFT_{region_clean}_SUVR",
            f"RIGHT_{region_clean}_SUVR",
            f"{region_clean}_SUVR",
        ]
        found = None
        for pattern in patterns:
            if pattern in df.columns:
                found = pattern
                break
        cols.append(found)
    return cols

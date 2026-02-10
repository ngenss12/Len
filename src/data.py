import os
import glob
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Candidate encodings for Baidu/AIS Studio dataset files
_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]


def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path} with encodings {_CANDIDATE_ENCODINGS}: {last_err}")


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", "", str(c)).lower()


def detect_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detect column names across possible variants (Chinese/English/BOM).
    Returns mapping: id, time, lon, lat, speed, course, label(optional).
    """
    cols = { _norm_col(c): c for c in df.columns }

    def pick(cands: List[str], required: bool=True) -> Optional[str]:
        for k in cands:
            k2 = _norm_col(k)
            if k2 in cols:
                return cols[k2]
        if required:
            raise KeyError(f"Cannot find any of columns {cands} in columns={list(df.columns)}")
        return None

    id_col = pick(["渔船ID", "渔船id", "id", "ID", "vessel_id", "ship_id"])
    time_col = pick(["time", "timestamp", "datetime", "时间", "t"], required=False)
    lon_col = pick(["lon", "longitude", "经度"])
    lat_col = pick(["lat", "latitude", "纬度"])
    speed_col = pick(["speed", "sog", "速度"], required=False)
    course_col = pick(["course", "cog", "航向", "direction", "dir"], required=False)
    label_col = pick(["type", "label", "y", "类别"], required=False)

    return {
        "id": id_col,
        "time": time_col,
        "lon": lon_col,
        "lat": lat_col,
        "speed": speed_col,
        "course": course_col,
        "label": label_col,
    }


LABEL_TO_ID = {
    # Chinese labels in AIS Studio dataset
    "围网": 0,  # encircling / seine
    "拖网": 1,  # trawler
    "刺网": 2,  # gillnet
    # English fallbacks
    "seine": 0,
    "encircling": 0,
    "trawler": 1,
    "trawl": 1,
    "gillnet": 2,
    "gill-net": 2,
}
ID_TO_LABEL_ZH = {0: "围网", 1: "拖网", 2: "刺网"}


def label_to_int(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s == "":
        return None
    # numeric string
    if re.fullmatch(r"\d+", s):
        return int(s)
    s_low = s.lower()
    return LABEL_TO_ID.get(s, LABEL_TO_ID.get(s_low, None))


@dataclass
class SequenceSample:
    vessel_id: int
    X: np.ndarray          # (T, F)
    y: Optional[int]       # class id 0/1/2 or None


def _find_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))


def load_baidu_folder(folder: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load a folder that may contain:
    - one or multiple trajectory csv files (with lon/lat etc)
    - optionally a label mapping file (with just 渔船ID + type)
    Returns:
        traj_df, label_df (optional)
    """
    csvs = _find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under: {folder}")

    traj_parts = []
    label_parts = []

    for p in csvs:
        df = read_csv_robust(p)
        # quick classify: if has lon/lat columns -> trajectory; else if has type -> label map
        cols_norm = set(_norm_col(c) for c in df.columns)
        has_lon = any(k in cols_norm for k in [_norm_col("lon"), _norm_col("longitude"), _norm_col("经度")])
        has_lat = any(k in cols_norm for k in [_norm_col("lat"), _norm_col("latitude"), _norm_col("纬度")])
        has_type = any(k in cols_norm for k in [_norm_col("type"), _norm_col("label"), _norm_col("类别")])
        if has_lon and has_lat:
            traj_parts.append(df)
        elif has_type:
            label_parts.append(df)

    if not traj_parts:
        # maybe all in one file but columns not detected due to weird names
        # fallback: treat all as traj
        traj_parts = [read_csv_robust(p) for p in csvs]
        label_parts = []

    traj_df = pd.concat(traj_parts, ignore_index=True)
    label_df = pd.concat(label_parts, ignore_index=True) if label_parts else None
    return traj_df, label_df


def build_samples_from_df(
    traj_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame],
    resample_mins: Optional[int] = None,
    max_len: Optional[int] = 512,
    require_label: bool = True,
) -> List[SequenceSample]:
    """
    Convert raw points into one sample per vessel_id:
      - sort by time if available
      - optionally resample at a fixed minutes interval
      - keep features [lon, lat, speed, course] (missing speed/course -> fill 0)
      - truncate/pad handled later in collate_fn; here we only truncate to max_len if set
    """
    schema = detect_schema(traj_df)
    id_col = schema["id"]
    time_col = schema["time"]
    lon_col = schema["lon"]
    lat_col = schema["lat"]
    speed_col = schema["speed"]
    course_col = schema["course"]
    label_col = schema["label"]

    # label mapping from label_df (if provided)
    id_to_label = None
    if label_df is not None:
        ls = detect_schema(label_df)
        lid = ls["id"]
        llab = ls["label"]
        if llab is None:
            # some label files may use "type" already caught, but just in case:
            raise KeyError("Label file found but cannot detect label column.")
        tmp = label_df[[lid, llab]].copy()
        tmp[lid] = pd.to_numeric(tmp[lid], errors="coerce")
        tmp = tmp.dropna(subset=[lid])
        tmp[lid] = tmp[lid].astype(int)
        tmp["y"] = tmp[llab].map(label_to_int)
        tmp = tmp.dropna(subset=["y"])
        id_to_label = dict(zip(tmp[lid].tolist(), tmp["y"].astype(int).tolist()))

    df = traj_df.copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df.dropna(subset=[id_col])
    df[id_col] = df[id_col].astype(int)

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

    # fill missing speed/course with 0
    if speed_col is None:
        df["_speed"] = 0.0
        speed_col = "_speed"
    else:
        df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce").fillna(0.0)

    if course_col is None:
        df["_course"] = 0.0
        course_col = "_course"
    else:
        df[course_col] = pd.to_numeric(df[course_col], errors="coerce").fillna(0.0)

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col])

    # bring label into df
    if label_col is not None:
        df["y"] = df[label_col].map(label_to_int)
    elif id_to_label is not None:
        df["y"] = df[id_col].map(id_to_label)
    else:
        df["y"] = None

    if require_label and df["y"].isna().all():
        raise ValueError(
            "No labels detected. Ensure train data contains a 'type' column or a separate label file exists."
        )

    samples: List[SequenceSample] = []

    for vid, g in df.groupby(id_col):
        g = g.copy()
        if time_col is not None:
            g = g.sort_values(time_col)

        # pick a single label for the vessel (most common non-null)
        y = None
        if "y" in g.columns:
            ys = g["y"].dropna().astype(int)
            if len(ys) > 0:
                y = int(ys.value_counts().index[0])

        # optional resampling to fixed interval
        if resample_mins is not None and time_col is not None:
            g = g.set_index(time_col)
            # numeric columns to resample
            g2 = g[[lon_col, lat_col, speed_col, course_col]].resample(f"{resample_mins}min").mean()
            g2 = g2.interpolate(limit_direction="both")
            g2 = g2.reset_index()
            # restore y (unchanged)
            g = g2

        X = g[[lon_col, lat_col, speed_col, course_col]].to_numpy(dtype=np.float32)

        if max_len is not None and X.shape[0] > max_len:
            # keep the most recent max_len points (often best for behavior)
            X = X[-max_len:, :]

        samples.append(SequenceSample(vessel_id=int(vid), X=X, y=y))

    return samples

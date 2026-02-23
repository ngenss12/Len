import argparse
import base64
from io import BytesIO
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from folium.raster_layers import ImageOverlay
import tifffile as tiff
import matplotlib.pyplot as plt


def choose_coord_cols(df: pd.DataFrame):
    candidates = [
        ("Center_latitude", "Center_longitude"),
        ("AIS_Latitude", "AIS_Longitude"),
        ("Projected_Latitude", "Projected_Longitude"),
    ]
    for lat_col, lon_col in candidates:
        if lat_col in df.columns and lon_col in df.columns:
            return lat_col, lon_col
    raise KeyError(
        "No coordinate columns found. Need one of: "
        "(Center_latitude, Center_longitude), "
        "(AIS_Latitude, AIS_Longitude), "
        "(Projected_Latitude, Projected_Longitude)."
    )


def load_and_match(metadata_csv: Path, image_dir: Path, name_col: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if name_col not in df.columns:
        raise KeyError(f"Column '{name_col}' not found in {metadata_csv}")

    df = df.copy()
    df["file_name"] = df[name_col].astype(str).apply(lambda s: Path(s).name)

    image_files = list(image_dir.rglob("*.tif")) + list(image_dir.rglob("*.tiff"))
    img_df = pd.DataFrame(
        {
            "file_name": [p.name for p in image_files],
            "img_path": [str(p) for p in image_files],
        }
    )

    joined = df.merge(img_df, on="file_name", how="left")
    return joined


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    # Accept (H,W), (H,W,C), or (C,H,W). Return (H,W,3) uint8.
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 2:
        z = np.zeros_like(img[..., :1])
        img = np.concatenate([img, z], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255.0).astype(np.uint8)


def _meters_to_latlon_delta(meters: float, lat_deg: float):
    # Approx conversion suitable for small overlays.
    dlat = meters / 111_320.0
    dlon = meters / (111_320.0 * np.cos(np.deg2rad(lat_deg)).clip(min=1e-6))
    return float(dlat), float(dlon)


def _thumb_data_uri_from_tiff(img_path: str, width_px: int = 180):
    """
    Build data URI PNG thumbnail from TIFF so popup can display image inline.
    Returns None if conversion fails.
    """
    try:
        img = tiff.imread(img_path)
        rgb = _to_rgb_uint8(np.asarray(img))
        h, w = rgb.shape[:2]
        if w <= 0 or h <= 0:
            return None

        # Preserve aspect ratio.
        new_w = max(1, int(width_px))
        new_h = max(1, int(round(h * (new_w / w))))

        fig = plt.figure(figsize=(new_w / 100.0, new_h / 100.0), dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(rgb)
        ax.axis("off")

        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default="metadata.csv")
    ap.add_argument("--image_dir", default="resized_new")
    ap.add_argument("--name_col", default="patch_cal", help="Metadata column holding image name/path")
    ap.add_argument("--out_html", default="outputs/google_earth_tiff_map.html")
    ap.add_argument("--max_points", type=int, default=0, help="0 = use all points")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cluster", action="store_true", default=True)
    ap.add_argument("--no-cluster", dest="cluster", action="store_false")
    ap.add_argument("--popup_image", action="store_true", default=True, help="Show TIFF thumbnail in point popup")
    ap.add_argument("--no-popup_image", dest="popup_image", action="store_false")
    ap.add_argument("--popup_image_width", type=int, default=180)
    ap.add_argument("--overlay_images", action="store_true", default=False, help="Overlay TIFF patches on map (approximate bounds)")
    ap.add_argument("--max_overlays", type=int, default=100, help="Max image overlays (use small number for performance)")
    ap.add_argument("--overlay_size_m", type=float, default=300.0, help="Approx side length of each overlay in meters")
    args = ap.parse_args()

    metadata_csv = Path(args.metadata_csv)
    image_dir = Path(args.image_dir)
    out_html = Path(args.out_html)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata csv not found: {metadata_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")

    df = load_and_match(metadata_csv, image_dir, args.name_col)
    lat_col, lon_col = choose_coord_cols(df)

    # Keep only matched files with valid coordinates
    df = df.dropna(subset=["img_path", lat_col, lon_col]).copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()

    if len(df) == 0:
        raise RuntimeError("No matched rows with valid coordinates.")

    if args.max_points and len(df) > args.max_points:
        df = df.sample(args.max_points, random_state=args.seed).copy()

    center_lat = float(df[lat_col].median())
    center_lon = float(df[lon_col].median())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None, control_scale=True)

    # Google Earth-like layers
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Hybrid",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)

    container = MarkerCluster(name="TIFF points").add_to(m) if args.cluster else m

    thumb_cache = {}
    for _, row in df.iterrows():
        category = str(row["category"]) if "category" in df.columns else "-"
        file_name = row["file_name"]
        img_path = row["img_path"]

        thumb_html = ""
        if args.popup_image:
            if img_path not in thumb_cache:
                thumb_cache[img_path] = _thumb_data_uri_from_tiff(img_path, width_px=args.popup_image_width)
            data_uri = thumb_cache[img_path]
            if data_uri:
                thumb_html = f'<img src="{data_uri}" style="width:{args.popup_image_width}px; height:auto; border:1px solid #ccc;"><br>'

        popup_html = (
            f"{thumb_html}"
            f"<b>file:</b> {file_name}<br>"
            f"<b>category:</b> {category}<br>"
            f"<b>lat:</b> {row[lat_col]:.6f}<br>"
            f"<b>lon:</b> {row[lon_col]:.6f}<br>"
            f"<b>img_path:</b> {img_path}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            color="#00bcd4",
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=450),
            tooltip=file_name,
        ).add_to(container)

    folium.LayerControl(collapsed=False).add_to(m)

    if args.overlay_images:
        if len(df) > args.max_overlays:
            df_overlay = df.sample(args.max_overlays, random_state=args.seed).copy()
        else:
            df_overlay = df

        overlay_group = folium.FeatureGroup(name="TIFF overlays (approx)", show=False)
        for _, row in df_overlay.iterrows():
            try:
                img = tiff.imread(row["img_path"])
                rgb = _to_rgb_uint8(np.asarray(img))
                dlat, dlon = _meters_to_latlon_delta(args.overlay_size_m / 2.0, float(row[lat_col]))
                bounds = [
                    [float(row[lat_col]) - dlat, float(row[lon_col]) - dlon],  # south-west
                    [float(row[lat_col]) + dlat, float(row[lon_col]) + dlon],  # north-east
                ]
                ImageOverlay(
                    image=rgb,
                    bounds=bounds,
                    opacity=0.55,
                    interactive=False,
                    cross_origin=False,
                    zindex=2,
                ).add_to(overlay_group)
            except Exception:
                # Skip broken/unreadable images without stopping map generation.
                continue
        overlay_group.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))

    print(f"Saved map: {out_html}")
    print(f"Points plotted: {len(df)}")
    print(f"Coordinates used: {lat_col}, {lon_col}")
    if args.overlay_images:
        print(f"Overlays enabled (approximate), max_overlays={args.max_overlays}, overlay_size_m={args.overlay_size_m}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default="metadata.csv")
    ap.add_argument("--image_dir", default="resized_new")
    ap.add_argument("--meta_name_col", default="patch_cal", help="Column in metadata that stores TIFF filename/path")
    ap.add_argument("--out_join_csv", default="final/metadata_image_join.csv")
    ap.add_argument("--out_missing_in_meta_csv", default="final/missing_in_metadata.csv")
    ap.add_argument("--out_missing_in_image_csv", default="final/missing_in_images.csv")
    ap.add_argument("--out_dup_meta_csv", default="final/duplicate_names_in_metadata.csv")
    args = ap.parse_args()

    meta_path = Path(args.metadata_csv)
    img_dir = Path(args.image_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    meta = pd.read_csv(meta_path)
    if args.meta_name_col not in meta.columns:
        raise KeyError(f"Column '{args.meta_name_col}' not found in metadata.csv")

    # Normalize metadata filename to basename only
    meta = meta.copy()
    meta["file_name"] = meta[args.meta_name_col].astype(str).apply(lambda s: Path(s).name)

    # Build image dataframe
    img_files = sorted(list(img_dir.rglob("*.tif")) + list(img_dir.rglob("*.tiff")))
    img_df = pd.DataFrame(
        {
            "file_name": [p.name for p in img_files],
            "img_path": [str(p) for p in img_files],
        }
    )

    # Duplicate checks
    dup_meta = meta[meta.duplicated("file_name", keep=False)].sort_values("file_name")
    dup_img = img_df[img_df.duplicated("file_name", keep=False)].sort_values("file_name")
    if len(dup_img) > 0:
        print("[warn] Duplicate file names found in image folder. Join may be ambiguous.")

    # Left join metadata -> images
    joined = meta.merge(img_df, on="file_name", how="left", indicator=True)
    missing_in_images = joined[joined["_merge"] == "left_only"].copy()

    # Missing in metadata: images that have no metadata row
    meta_names = set(meta["file_name"].unique())
    img_names = set(img_df["file_name"].unique())
    missing_meta_names = sorted(img_names - meta_names)
    missing_in_meta = img_df[img_df["file_name"].isin(missing_meta_names)].copy()

    # Save outputs
    out_join = Path(args.out_join_csv)
    out_join.parent.mkdir(parents=True, exist_ok=True)
    joined.drop(columns=["_merge"]).to_csv(out_join, index=False, encoding="utf-8-sig")

    Path(args.out_missing_in_meta_csv).parent.mkdir(parents=True, exist_ok=True)
    missing_in_meta.to_csv(args.out_missing_in_meta_csv, index=False, encoding="utf-8-sig")
    missing_in_images.to_csv(args.out_missing_in_image_csv, index=False, encoding="utf-8-sig")
    dup_meta.to_csv(args.out_dup_meta_csv, index=False, encoding="utf-8-sig")

    print("=== Metadata-TIFF Match Summary ===")
    print(f"metadata_rows: {len(meta)}")
    print(f"metadata_unique_names: {meta['file_name'].nunique()}")
    print(f"image_files: {len(img_df)}")
    print(f"image_unique_names: {img_df['file_name'].nunique()}")
    print(f"missing_in_images (metadata without file): {len(missing_in_images)}")
    print(f"missing_in_metadata (file without metadata): {len(missing_in_meta)}")
    print(f"duplicate_names_in_metadata: {len(dup_meta)}")
    print(f"duplicate_names_in_images: {len(dup_img)}")
    print(f"saved_join_csv: {out_join}")
    print(f"saved_missing_in_metadata_csv: {args.out_missing_in_meta_csv}")
    print(f"saved_missing_in_images_csv: {args.out_missing_in_image_csv}")
    print(f"saved_duplicate_meta_csv: {args.out_dup_meta_csv}")


if __name__ == "__main__":
    main()

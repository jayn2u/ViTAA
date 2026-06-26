import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from vitaa.utils.paths import (
    TEXT_ATTRIBUTE_GRAPH_DRIVE_ID,
    cuhk_pedes_paths,
    load_project_env,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _has_attribute_graph_jsons(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.json"))


def _normalize_attribute_graph_dir(vitaa_data_root: Path, graph_dir: Path) -> None:
    if _has_attribute_graph_jsons(graph_dir):
        return
    nested_dirs = [path for path in vitaa_data_root.iterdir() if path.is_dir()]
    for nested in nested_dirs:
        if _has_attribute_graph_jsons(nested):
            if graph_dir.exists():
                shutil.rmtree(graph_dir)
            nested.rename(graph_dir)
            return
    raise FileNotFoundError(
        f"No attribute graph JSON files found under {vitaa_data_root}."
    )


def _extract_archive(archive: Path, vitaa_data_root: Path, graph_dir: Path) -> None:
    vitaa_data_root.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(vitaa_data_root)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive}")
    _normalize_attribute_graph_dir(vitaa_data_root, graph_dir)


def ensure_text_attribute_graph(paths: dict) -> None:
    graph_dir = paths["text_attribute_graph_dir"]
    archive = paths["text_attribute_graph_archive"]
    if _has_attribute_graph_jsons(graph_dir):
        print(f"text_attribute_graph already present: {graph_dir}")
        return

    vitaa_data_root = paths["vitaa_data_root"]
    vitaa_data_root.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        _run(
            [
                "uv",
                "run",
                "gdown",
                TEXT_ATTRIBUTE_GRAPH_DRIVE_ID,
                "-O",
                str(archive),
            ]
        )
    _extract_archive(archive, vitaa_data_root, graph_dir)
    print(f"text_attribute_graph ready: {graph_dir}")


def ensure_converted_annotations(paths: dict) -> None:
    annotations_dir = paths["annotations_dir"]
    required = [annotations_dir / name for name in ("train.json", "val.json", "test.json")]
    if all(path.is_file() for path in required):
        print(f"converted annotations already present: {annotations_dir}")
        return

    _run(
        [
            "uv",
            "run",
            "python",
            "tools/cuhkpedes/convert_to_json.py",
            "--datadir",
            str(paths["vitaa_data_root"]),
            "--outdir",
            str(annotations_dir),
            "--reid-raw",
            str(paths["reid_raw"]),
        ]
    )


def validate_raw_dataset(paths: dict) -> None:
    img_dir = paths["img_dir"]
    reid_raw = paths["reid_raw"]
    if not img_dir.is_dir():
        raise FileNotFoundError(f"CUHK-PEDES images not found: {img_dir}")
    if not reid_raw.is_file():
        raise FileNotFoundError(f"CUHK-PEDES annotation not found: {reid_raw}")


def validate_segs(paths: dict) -> None:
    segs_dir = paths["segs_dir"]
    if not segs_dir.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory missing: {segs_dir}. "
            "Generate segs with Human-Parsing-Network before training."
        )
    seg_count = sum(1 for _ in segs_dir.rglob("*.png"))
    if seg_count == 0:
        raise FileNotFoundError(
            f"No segmentation PNG files found under {segs_dir}."
        )
    print(f"segs ready: {segs_dir} ({seg_count} png files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ViTAA prerequisites from env/.env")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip gdown when text_attribute_graph is already extracted",
    )
    parser.add_argument(
        "--require-segs",
        action="store_true",
        help="Fail if segs/ is missing (default: warn only)",
    )
    args = parser.parse_args()

    load_project_env()
    paths = cuhk_pedes_paths()
    validate_raw_dataset(paths)

    paths["vitaa_data_root"].mkdir(parents=True, exist_ok=True)
    paths["annotations_dir"].mkdir(parents=True, exist_ok=True)
    paths["segs_dir"].mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ensure_text_attribute_graph(paths)
    elif not _has_attribute_graph_jsons(paths["text_attribute_graph_dir"]):
        raise FileNotFoundError(
            f"text_attribute_graph missing under {paths['text_attribute_graph_dir']}"
        )

    ensure_converted_annotations(paths)

    try:
        validate_segs(paths)
    except FileNotFoundError as exc:
        if args.require_segs:
            raise
        print(f"WARNING: {exc}")

    print("Prerequisites complete up to training (except segs if warned above).")


if __name__ == "__main__":
    main()

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env"
CUHK_PEDES_DIRNAME = "CUHK-PEDES"
TEXT_ATTRIBUTE_GRAPH_ARCHIVE = "text_attribute_graph.zip"
TEXT_ATTRIBUTE_GRAPH_DRIVE_ID = "1Sqm3V97hbqK9GxIwshZejJWLARfu5o1s"


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


def load_project_env() -> None:
    for key, value in _read_env_file(ENV_FILE).items():
        os.environ.setdefault(key, value)


def _require_env(key: str) -> str:
    load_project_env()
    value = os.environ.get(key, "").strip()
    if value:
        return value
    raise RuntimeError(
        f"{key} is not set. Add it to {ENV_FILE} or export it before running."
    )


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_dataset_root() -> Path:
    return _resolve_path(_require_env("DATASET_ROOT"))


def resolve_vitaa_data_root() -> Path:
    load_project_env()
    value = os.environ.get("VITAA_DATA_ROOT", "").strip()
    if value:
        return _resolve_path(value)
    return resolve_dataset_root() / CUHK_PEDES_DIRNAME / "vitaa"


def cuhk_pedes_paths():
    dataset_root = resolve_dataset_root()
    vitaa_data_root = resolve_vitaa_data_root()
    cuhk_root = dataset_root / CUHK_PEDES_DIRNAME
    return {
        "dataset_root": dataset_root,
        "vitaa_data_root": vitaa_data_root,
        "cuhk_root": cuhk_root,
        "img_dir": cuhk_root / "imgs",
        "reid_raw": cuhk_root / "reid_raw.json",
        "annotations_dir": vitaa_data_root / "annotations",
        "text_attribute_graph_dir": vitaa_data_root / "text_attribute_graph",
        "text_attribute_graph_archive": vitaa_data_root / TEXT_ATTRIBUTE_GRAPH_ARCHIVE,
        "segs_dir": vitaa_data_root / "segs",
    }

"""Python config loader and data-path resolver.

Experiment configs are Python modules with attribute access (e.g.
`config.gamma`, `config.rho_values`); sweep grids are tuples. The
preprocessing config `configs/data_process.py` is dict-based.
"""
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Union


PathLike = Union[str, Path]

ALLOWED_DATASETS = ("adni", "nacc", "adni_nacc_union")


def load_config(config_path: PathLike) -> ModuleType:
    """Load a Python config module by file path using importlib.util.

    Args:
      config_path: path to a .py file.

    Returns:
      The loaded module object. Attributes are accessed as
      config.method, config.gamma, config.rho_values, etc.

    Use a unique module name derived from the path so multiple configs
    can be loaded in the same Python process without collision in
    sys.modules.

    Raise FileNotFoundError if config_path does not exist.
    Raise an informative error if the file fails to import.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location(
        f"_loaded_config_{path.stem}_pending", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for config: {path}")

    module_name = f"_loaded_config_{path.stem}_{id(spec)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for config: {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Failed to import config '{path}': {exc.__class__.__name__}: {exc}"
        ) from exc
    return module


def resolve_data_path(config: ModuleType) -> Path:
    """Resolve the processed_data.npz path for a config.

    If `config.processed_data_path` is set, that file is used. Otherwise:
      data/processed/candidate_groups/<dataset_name>/
        ("normalization" | "raw")/processed_data.npz

    Raises ValueError for an unknown dataset_name and FileNotFoundError
    if the resolved file does not exist.
    """
    explicit = getattr(config, "processed_data_path", None)
    if explicit:
        resolved = Path(explicit)
        if not resolved.is_file():
            raise FileNotFoundError(f"processed_data_path {resolved} does not exist")
        return resolved

    dataset_name = getattr(config, "dataset_name")
    use_normalization = bool(getattr(config, "use_normalization"))

    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(
            f"Invalid dataset_name {dataset_name!r}; "
            f"allowed values: {ALLOWED_DATASETS}"
        )

    variant = "normalization" if use_normalization else "raw"
    resolved = (
        Path("data")
        / "processed"
        / "candidate_groups"
        / dataset_name
        / variant
        / "processed_data.npz"
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Processed data not found at {resolved}. "
            f"Run the preprocessing pipeline first."
        )
    return resolved


# ---- workdir-name-priority overrides for runtime values --------------------
#
# A config_snapshot.py inherits the defaults of configs/base.py (e.g.
# gamma=1000). Values set on the command line are recovered from the
# workdir name, which is treated as the source of truth.

_GAMMA_RE = re.compile(r"_gamma(\d+)")
_RHO_RE = re.compile(r"_rho([\d.]+)_")

_SUFFIX_TABLE = (
    ("_normalization_age", True),
    ("_normalization", True),
    ("_raw_age", False),
    ("_raw", False),
)


def apply_workdir_overrides(config: ModuleType, workdir: PathLike) -> ModuleType:
    """Override runtime values on `config` from the workdir name.

    Sets when detectable:
      - suffix, use_normalization   from the name suffix
      - gamma                       from a `_gammaN` token
      - rho                         from a `_rhoX_` token

    Idempotent. Returns config (mutated in place).
    """
    name = Path(workdir).resolve().name

    # Suffix / use_normalization (longest-match first)
    for suf, un in _SUFFIX_TABLE:
        if name.endswith(suf):
            config.suffix = suf
            config.use_normalization = un
            break

    # gamma
    m = _GAMMA_RE.search(name)
    if m:
        config.gamma = int(m.group(1))

    # rho
    m = _RHO_RE.search(name)
    if m:
        try:
            config.rho = float(m.group(1))
        except ValueError:
            pass

    return config

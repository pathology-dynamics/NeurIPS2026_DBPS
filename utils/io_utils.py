"""File-system helpers used by training and sampling pipelines."""
import re
import shutil
from pathlib import Path
from types import ModuleType
from typing import Union


PathLike = Union[str, Path]


_RESOLVED_BLOCK_START = "# ---- resolved runtime overrides ----"
_RESOLVED_BLOCK_END = "# ---- end resolved runtime overrides ----"
_RESOLVED_FIELDS = (
    "gamma",
    "rho",
    "dataset_name",
    "use_normalization",
    "suffix",
    "beta",
    "batch_size",
    "lr",
    "weight_decay",
    "epochs",
    "num_checkpoints",
    "num_candidate_sample",
)


def _format_resolved_block(config_module: ModuleType) -> str:
    """Build the `# ---- resolved runtime overrides ----` text block.

    Captures runtime-resolved values for fields commonly overridden via
    env / CLI. Uses repr() so strings get quoted, bools stay True/False,
    floats stay precise.
    """
    lines = [_RESOLVED_BLOCK_START]
    for field in _RESOLVED_FIELDS:
        if hasattr(config_module, field):
            val = getattr(config_module, field)
            try:
                lines.append(f"{field} = {val!r}")
            except Exception:
                # Skip un-reprable values (e.g. functions). Shouldn't happen
                # for the scalar fields above, but be defensive.
                continue
    lines.append(_RESOLVED_BLOCK_END)
    return "\n".join(lines)


def _write_resolved_block_idempotent(dest: Path, block: str) -> None:
    """Append OR replace the resolved block in dest. Idempotent."""
    text = dest.read_text()
    pattern = re.compile(
        re.escape(_RESOLVED_BLOCK_START) + r".*?" + re.escape(_RESOLVED_BLOCK_END),
        flags=re.S,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        # Append, ensuring exactly one blank line before the block.
        text = text.rstrip("\n") + "\n\n\n" + block + "\n"
    dest.write_text(text)


def ensure_dir(path: PathLike) -> Path:
    """Create directory and all parents if they do not exist.

    Returns the resolved Path object. Idempotent.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_config_snapshot(config_module: ModuleType, dest_dir: PathLike) -> Path:
    """Copy the .py source of `config_module` to `dest_dir/config_snapshot.py`,
    then append a `# ---- resolved runtime overrides ----` block recording
    the actual runtime values of gamma / rho / dataset_name /
    use_normalization / suffix / beta / batch_size / lr / weight_decay /
    epochs / num_checkpoints / num_candidate_sample.

    The block records values set on the command line, which a plain copy
    of the source file would lose. It is idempotent: if it already
    exists in dest, it is replaced rather than re-appended.

    Use config_module.__file__ to locate the source. Raise a clear
    RuntimeError if config_module has no __file__ attribute (e.g. it
    was constructed in memory rather than imported from a file).

    Returns the destination path.
    """
    source_path = getattr(config_module, "__file__", None)
    if source_path is None:
        raise RuntimeError(
            "save_config_snapshot: config_module has no __file__ attribute; "
            "cannot snapshot an in-memory module. Load the config from a "
            ".py file via utils.config_utils.load_config."
        )
    dest = ensure_dir(dest_dir) / "config_snapshot.py"
    shutil.copyfile(source_path, dest)
    block = _format_resolved_block(config_module)
    _write_resolved_block_idempotent(dest, block)
    return dest


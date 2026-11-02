"""Find an automation, script or scene in the YAML Home Assistant loads, by following
configuration.yaml: its includes, and its packages. Tags stay inert (load_inert), so no
secret, environment variable or file outside the config directory is ever resolved
into the result. Blocking: run it in an executor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from .util import Tagged, load_inert

MAX_FILES = 400
MAX_DEPTH = 6
_DIR_TAGS = (
    "!include_dir_list",
    "!include_dir_named",
    "!include_dir_merge_list",
    "!include_dir_merge_named",
)

# (file relative to the config directory, its data, the folder its own includes start from)
Found = tuple[str, Any, Path]


class _Walk:
    def __init__(self, root: Path, is_sensitive: Callable[[str], bool]) -> None:
        self.root = root.resolve()
        self.is_sensitive = is_sensitive
        self.read = 0

    def rel(self, path: Path) -> str | None:
        try:
            rel = path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return None  # outside the config directory: not followed
        return None if self.is_sensitive(rel) else rel

    def load(self, path: Path) -> tuple[str, Any] | None:
        rel = self.rel(path)
        if rel is None or self.read >= MAX_FILES:
            return None
        self.read += 1
        try:
            return rel, load_inert(path.read_text(encoding="utf-8"))
        except OSError, UnicodeDecodeError, yaml.YAMLError:
            return None

    def expand(self, base: Path, value: Any, rel: str, depth: int = 0) -> Iterator[Found]:
        """The value as Home Assistant reads it: itself, or what its include tag pulls
        in, one file at a time. A !include_dir_named file comes back as {stem: data},
        the way it lands in the merged result."""
        if not isinstance(value, Tagged):
            yield rel, value, base
            return
        if depth > MAX_DEPTH:
            return
        target = base / value.raw
        if self.rel(target) is None:
            return  # outside the config directory (or protected): not even listed
        if value.tag == "!include":
            paths = [target] if target.is_file() else []
        elif value.tag in _DIR_TAGS and target.is_dir():
            # every .yaml below the folder, in name order, hidden ones skipped
            paths = sorted(
                p
                for p in target.rglob("*.yaml")
                if not any(part.startswith(".") for part in p.relative_to(target).parts)
            )
        else:
            return  # !secret, !env_var...: never resolved
        for path in paths:
            loaded = self.load(path)
            if loaded is None:
                continue
            name, data = loaded
            if value.tag == "!include_dir_named":
                data = {path.stem: data}
            if isinstance(data, Tagged):
                yield from self.expand(path.parent, data, name, depth + 1)
            else:
                yield name, data, path.parent


def _blocks(walk: _Walk, found: Found, kind: str, depth: int) -> Iterator[tuple[str, Any]]:
    """(file, block) for each `kind` block of one config (`automation`, `automation
    garden`...), and for those of its packages."""
    rel, config, base = found
    if not isinstance(config, dict) or depth > MAX_DEPTH:
        return
    for key, value in config.items():
        key = str(key)
        if key == kind or key.startswith(f"{kind} "):
            for name, data, _folder in walk.expand(base, value, rel):
                yield name, data
        elif key == "homeassistant" and isinstance(value, dict) and "packages" in value:
            # packages: {name: config}, from the file itself or from what it includes
            for name, packages, folder in walk.expand(base, value["packages"], rel):
                if not isinstance(packages, dict):
                    continue
                for package in packages.values():
                    for inner in walk.expand(folder, package, name):
                        yield from _blocks(walk, inner, kind, depth + 1)


def find_in_yaml(
    root: Path, kind: str, item_id: str, is_sensitive: Callable[[str], bool]
) -> tuple[str, Any] | None:
    """The file (relative to root) and config of the item with this id, or None."""
    walk = _Walk(root, is_sensitive)
    loaded = walk.load(root / "configuration.yaml")
    if loaded is None:
        return None
    for rel, block in _blocks(walk, (loaded[0], loaded[1], walk.root), kind, 0):
        if kind == "script":
            if isinstance(block, dict) and isinstance(block.get(item_id), dict):
                return rel, block[item_id]
            continue
        for item in block if isinstance(block, list) else [block]:
            if isinstance(item, dict) and "id" in item and str(item["id"]) == str(item_id):
                return rel, item
    return None

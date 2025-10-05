import importlib
from pathlib import Path

from rootutils import find_root

root_dir: Path = find_root()
src_dir: Path = root_dir / "src"


def import_modules(pkg_dir: Path) -> None:
    """Import all Python modules in a package directory.

    Recursively finds all Python files in the given directory and imports them
    as modules. This is useful for ensuring all modules are loaded when the
    package is imported.

    Args:
        pkg_dir: Directory containing Python modules to import
    """
    modules: list[str] = python_modules(pkg_dir)
    for module in modules:
        importlib.import_module(name=module)


def python_modules(pkg_dir: Path) -> list[str]:
    """Get a list of module names for all Python files in a directory.

    Recursively searches for all .py files in the directory and converts
    their paths to module names, excluding __init__.py files.

    Args:
        pkg_dir: Directory to search for Python files

    Returns:
        List of module names (e.g., ['numerai.common.data.datamodule'])
    """
    return [module_name(path) for path in pkg_dir.rglob("*.py") if path.stem != "__init__"]


def module_name(module_path: Path) -> str:
    """Convert a file path to a module name.

    Converts a Python file path relative to the src directory into a
    dot-separated module name.

    Args:
        module_path: Path to the Python file

    Returns:
        Module name as a string (e.g., 'numerai.common.data.datamodule')
    """
    rel_path: Path = module_path.relative_to(src_dir)
    module_parts: list[str] = [*rel_path.parent.parts, rel_path.stem]
    return ".".join(module_parts)

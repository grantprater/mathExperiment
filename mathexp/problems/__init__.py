"""Problem families, by name."""

from .addfree import AddFree
from .base import Family, changed_fields, validate_mutation
from .setfam import SetFamily

FAMILIES: dict[str, Family] = {f.name: f for f in (SetFamily(), AddFree())}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES)}") from None


__all__ = ["FAMILIES", "Family", "get_family", "changed_fields", "validate_mutation"]

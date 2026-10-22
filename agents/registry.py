"""Agent registry with automatic discovery.

Every module in the ``agents`` package whose name ends in ``_agent`` is imported
on first use, so decorating a class with ``@register_agent`` is all it takes to
plug a new agent into the master agent, the weights editor, and the backtester.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from typing import TypeVar

from agents.base import BaseAgent

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseAgent]] = {}
_DISCOVERED = False
_LOCK = threading.Lock()

A = TypeVar("A", bound=type[BaseAgent])


def register_agent(cls: A) -> A:
    """Class decorator that registers an agent under its ``key``."""
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty 'key'")
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    existing = _REGISTRY.get(cls.key)
    if existing is not None and existing is not cls:
        # Re-import of the same module (e.g. Flask reloader) is fine; a real clash is not.
        if existing.__qualname__ != cls.__qualname__ or existing.__module__ != cls.__module__:
            raise ValueError(f"Agent key '{cls.key}' already registered by {existing.__name__}")
    _REGISTRY[cls.key] = cls
    return cls


def discover_agents() -> None:
    """Import every ``*_agent`` module in the agents package (idempotent)."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    with _LOCK:
        if _DISCOVERED:
            return
        import agents as package

        for module in pkgutil.iter_modules(package.__path__):
            if module.name.endswith("_agent") and module.name != "master_agent":
                try:
                    importlib.import_module(f"agents.{module.name}")
                except Exception:  # noqa: BLE001 - a broken plugin must not take down the app
                    logger.exception("Failed to import agent module agents.%s", module.name)
        _DISCOVERED = True


# Display order: stock-specific evidence first, market backdrop last.
CATEGORY_ORDER: tuple[str, ...] = ("Technical", "Flow", "Fundamental", "Sentiment", "Macro")


def _sort_key(cls: type[BaseAgent]) -> tuple[int, float, str]:
    try:
        rank = CATEGORY_ORDER.index(cls.category)
    except ValueError:
        rank = len(CATEGORY_ORDER)
    return rank, -cls.default_weight, cls.name


def get_agent_classes() -> dict[str, type[BaseAgent]]:
    """Return registered agent classes keyed by agent key, in display order."""
    discover_agents()
    return {cls.key: cls for cls in sorted(_REGISTRY.values(), key=_sort_key)}


def get_agent_class(key: str) -> type[BaseAgent]:
    classes = get_agent_classes()
    if key not in classes:
        raise KeyError(f"Unknown agent '{key}'. Registered: {sorted(classes)}")
    return classes[key]


def create_agents(keys: list[str] | None = None) -> list[BaseAgent]:
    """Instantiate agents by key (all registered agents when ``keys`` is None)."""
    classes = get_agent_classes()
    selected = keys if keys is not None else list(classes)
    return [get_agent_class(k)() for k in selected]


def agent_metadata() -> list[dict]:
    return [cls.metadata() for cls in get_agent_classes().values()]

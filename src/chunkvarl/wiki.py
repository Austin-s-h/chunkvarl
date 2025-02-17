"""OSRS Wiki integration for item icons."""

import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

try:
    from osrsreboxed import items_api

    HAS_OSRSBOX = True
except ImportError:
    HAS_OSRSBOX = False

logger = logging.getLogger(__name__)

# Cache for loaded items
_items_cache: dict[str, dict[str, Any]] = {}


def _load_items() -> None:
    """Load items from osrsreboxed if available."""
    global _items_cache
    if not HAS_OSRSBOX:
        return

    try:
        items = items_api.load()
        for item in items:
            _items_cache[item.name] = {"id": item.id, "icon": item.icon}
    except Exception as e:
        logger.error(f"Failed to load items from osrsreboxed: {e}")


def has_icon(item_id: str) -> bool:
    """Check if item has a wiki icon available."""
    if not _items_cache and HAS_OSRSBOX:
        _load_items()
    return item_id in _items_cache


def get_icon(item_id: str) -> Image.Image | None:
    """Fetch item icon.

    Args:
        item_id: The item identifier

    Returns:
        PIL Image if successful, None otherwise
    """
    if not has_icon(item_id):
        return None

    try:
        # Get base64 icon data
        icon_data = _items_cache[item_id]["icon"]
        if not icon_data:
            return None

        # Convert base64 to image
        icon_bytes = base64.b64decode(icon_data)
        return Image.open(BytesIO(icon_bytes))
    except Exception as e:
        logger.error(f"Failed to load icon for {item_id}: {e}")
        return None

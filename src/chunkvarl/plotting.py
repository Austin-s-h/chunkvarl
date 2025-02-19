from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt


class RarityTier(Enum):
    """OSRS item rarity tiers."""

    COMMON = ("common", "#45B6FE")  # Light Blue (instead of white)
    UNCOMMON = ("uncommon", "#00FF00")  # Green
    RARE = ("rare", "#0000FF")  # Blue
    VERY_RARE = ("very_rare", "#800080")  # Purple
    ULTRA_RARE = ("ultra_rare", "#FF0000")  # Red

    @property
    def value_name(self) -> str:
        """Get the string name of the rarity tier."""
        return self.value[0]

    @property
    def color(self) -> str:
        """Get the color associated with this rarity tier."""
        return self.value[1]


@dataclass(frozen=True)
class OsrsColors:
    background: str = "#F5DEB3"  # Wheat background
    panel: str = "#FFF8DC"  # Cornsilk panel
    border: str = "#8B7355"  # OSRS brown border
    text: str = "#2F2F2F"  # Dark grey text
    highlight: str = "#C17000"  # OSRS gold highlight
    gold: str = "#FFD700"  # Primary color - Gold
    blue: str = "#4682B4"  # Secondary color - Steel Blue
    bars: tuple[str, str] = ("#FFD700", "#4682B4")  # Gold and blue bars
    grid: str = "#DDDDDD"  # Light grey grid
    error_bars: str = "#8B4513"  # Dark brown for error bars
    actual_color: str = "#C17000"  # OSRS Gold color for actual bars
    dashed_lines: str = "#4B0082"  # Dark indigo for dashed lines

    def get_rarity_tier(self, drop_rate: float) -> RarityTier:
        """Get rarity tier based on drop rate."""
        if drop_rate > 1 / 10:  # > 1/10
            return RarityTier.COMMON
        elif drop_rate > 1 / 100:  # 1/10 - 1/100
            return RarityTier.UNCOMMON
        elif drop_rate > 1 / 1000:  # 1/100 - 1/1000
            return RarityTier.RARE
        elif drop_rate > 1 / 5000:  # 1/1000 - 1/5000
            return RarityTier.VERY_RARE
        else:  # ≤ 1/5000
            return RarityTier.ULTRA_RARE

    def get_rarity_color(self, drop_rate: float) -> str:
        """Get color based on OSRS Wiki rarity tiers."""
        return self.get_rarity_tier(drop_rate).color  # Return the actual rarity color


# Replace dictionary with dataclass instance
OSRS_COLORS = OsrsColors()


def apply_osrs_style() -> None:
    """Apply OSRS-inspired light theme styling to matplotlib plots."""
    plt.style.use("default")  # Start with light theme
    plt.rcParams.update(
        {
            "figure.facecolor": OSRS_COLORS.background,
            "axes.facecolor": OSRS_COLORS.panel,
            "axes.edgecolor": OSRS_COLORS.border,
            "axes.labelcolor": OSRS_COLORS.text,
            "text.color": OSRS_COLORS.text,
            "xtick.color": OSRS_COLORS.text,
            "ytick.color": OSRS_COLORS.text,
            "grid.color": OSRS_COLORS.grid,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titleweight": "bold",
            "savefig.facecolor": OSRS_COLORS.background,
            "figure.frameon": True,
        }
    )

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt
from tqdm import tqdm


@dataclass(frozen=True)
class OsrsColors:
    background: str = "#F5DEB3"  # Wheat background
    panel: str = "#FFF8DC"  # Cornsilk panel
    border: str = "#8B7355"  # OSRS brown border
    text: str = "#2F2F2F"  # Dark grey text
    highlight: str = "#C17000"  # OSRS gold highlight
    gold: str = "#FFD700"  # OSRS gold
    bars: tuple[str, str] = (
        "#FFD700",
        "#4682B4",
    )  # Gold and blue (colorblind-friendly)
    grid: str = "#DDDDDD"  # Light grey grid
    error_bars: str = "#8B4513"  # Dark brown for error bars


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


class Item(BaseModel):
    item_id: str
    drop_rate: NonNegativeFloat = Field(..., le=1.0)  # Must be between 0 and 1
    received: NonNegativeInt


class ItemCollection(BaseModel):
    items: list[Item]


def read_data(file_path: str | Path) -> pd.DataFrame:
    """Read data from a CSV file and validate using Pydantic."""
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file_path)

        # Convert DataFrame to list of dictionaries
        items_data = df.to_dict("records")

        # Validate with Pydantic
        validated_data = ItemCollection(
            items=[Item(**{str(k): v for k, v in item.items()}) for item in items_data]
        )

        # Convert back to DataFrame
        return pd.DataFrame([item.model_dump() for item in validated_data.items])

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file: {file_path}")
    except Exception as e:
        raise ValueError(f"Error validating data: {e!s}")


def simulate_searches(df: pd.DataFrame, seed: int) -> int:
    """Simulate searches until all items are found.

    Args:
        df: DataFrame containing item data
        seed: Random seed for reproducibility

    Returns:
        Number of searches required
    """
    np.random.seed(seed)
    searches: int = 1  # Start at 1 to ensure we never return 0
    items_collected: dict[str, int] = {item_id: 0 for item_id in df["item_id"]}
    while any(
        items_collected[item_id] < received
        for item_id, received in zip(df["item_id"], df["received"])
    ):
        for item_id, drop_rate in zip(df["item_id"], df["drop_rate"]):
            if np.random.random() < drop_rate:
                items_collected[item_id] += 1
                break
        searches += 1
    return searches


def calculate_dryness_for_item(
    item_id: str, drop_rate: float, num_simulations: int
) -> tuple[str, float]:
    """Calculate dryness for a single item through simulation.

    Args:
        item_id: Identifier for the item
        drop_rate: Drop rate as a probability
        num_simulations: Number of simulations to run

    Returns:
        Tuple of (item_id, average_searches)
    """
    searches_list: list[int] = []
    for _ in range(num_simulations):
        searches = 0
        found = False
        while not found:
            searches += 1
            if np.random.random() < drop_rate:
                found = True
        searches_list.append(searches)
    return item_id, float(np.mean(searches_list))


def _simulate_search_wrapper(params: tuple[pd.DataFrame, int]) -> int:
    """Wrapper function for multiprocessing."""
    df, seed = params
    return simulate_searches(df, seed)


def simulate_searches_multi(df: pd.DataFrame, num_simulations: int) -> list[int]:
    """Run multiple search simulations in parallel."""
    with mp.Pool(mp.cpu_count()) as pool:
        seeds = np.random.randint(0, int(1e8), size=num_simulations)
        params = [(df, seed) for seed in seeds]
        results = list(
            tqdm(
                pool.imap(_simulate_search_wrapper, params),
                total=num_simulations,
                desc="Running simulations",
            )
        )
    return results


def calculate_dryness(
    df: pd.DataFrame, total_searches: float
) -> str | dict[str, float]:
    """Calculate dryness for items with 0 drops.

    Args:
        df: DataFrame containing item data
        total_searches: Total number of searches performed

    Returns:
        Dictionary of dryness values or message string if no dry items

    Raises:
        ValueError: If DataFrame is empty
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    dry_items = df[df["received"] == 0]
    if dry_items.empty:
        return "No items with a value of 0 in your data."

    results = {}
    for _, row in dry_items.iterrows():
        # Expected number of drops after total_searches attempts
        # expected_drops = total_searches * row["drop_rate"]
        # Probability of getting 0 drops in total_searches attempts
        prob_zero = (1 - row["drop_rate"]) ** total_searches
        # Times over expected rate (using probability)
        times_over_rate = -np.log(prob_zero) if prob_zero > 0 else float("inf")
        results[row["item_id"]] = times_over_rate

    return results


def _bootstrap_wrapper(params: tuple[pd.DataFrame, int]) -> NDArray[np.float64]:
    """Wrapper function for bootstrap sampling."""
    df, _ = params
    return np.random.choice(df["received"], size=len(df), replace=True)


def calculate_bootstrap_errors(
    df: pd.DataFrame, num_bootstrap: int = 1000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate bootstrap confidence intervals for expected drops."""
    with mp.Pool(mp.cpu_count()) as pool:
        params = [(df, i) for i in range(num_bootstrap)]
        bootstrap_results = list(
            tqdm(
                pool.imap(_bootstrap_wrapper, params),
                total=num_bootstrap,
                desc="Calculating bootstrap intervals",
            )
        )

    bootstrap_array = np.array(
        [
            total_searches * df["drop_rate"]
            for bootstrap_sample in bootstrap_results
            for total_searches in [bootstrap_sample.sum() / np.sum(df["drop_rate"])]
        ]
    )

    lower_ci = np.percentile(bootstrap_array, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_array, 97.5, axis=0)

    return lower_ci, upper_ci


def plot_results(
    df: pd.DataFrame,
    simulated_searches: list[int],
    total_searches: float,
    dryness_results: dict[str, float],
    output_dir: Path | str = "plots",
) -> None:
    """Generate and save analysis plots as a single combined figure."""
    plots_dir = Path(output_dir)
    try:
        plots_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        raise PermissionError(f"Cannot create plots directory at {plots_dir}")

    apply_osrs_style()

    # Calculate figure size for 9:16 aspect ratio
    width = 16  # inches
    height = width * (16 / 9)  # maintains 9:16 ratio

    # Create figure with GridSpec
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # Increase default font sizes
    plt.rc("font", size=18)  # controls default text size
    plt.rc("axes", titlesize=24)  # fontsize of the title
    plt.rc("axes", labelsize=18)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=18)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=18)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=18)  # fontsize of the legend

    # Create subplots
    ax1 = fig.add_subplot(gs[0, :])  # Expected vs Actual (spans both columns)
    ax2 = fig.add_subplot(gs[1, 0])  # Search Distribution
    ax3 = fig.add_subplot(gs[1, 1])  # Dryness Analysis

    # Expected vs Actual Plot (ax1)
    ax1.set_facecolor(OSRS_COLORS.panel)
    x = np.arange(len(df))
    expected_drops = total_searches * df["drop_rate"]
    lower_ci, upper_ci = calculate_bootstrap_errors(df)

    ax1.bar(
        x - 0.2,
        expected_drops,
        width=0.4,
        label="Expected",
        color=OSRS_COLORS.gold,
        alpha=0.9,
    )
    for i, v in enumerate(expected_drops):
        ax1.text(
            x[i] - 0.2,
            v + 0.5,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            color=OSRS_COLORS.text,
            fontweight="bold",
            fontsize=16,
        )

    ax1.bar(
        x + 0.2,
        df["received"],
        width=0.4,
        label="Actual",
        color=OSRS_COLORS.bars[1],
        alpha=0.9,
    )

    # Error bars logic for non-zero values
    zero_mask = df["received"] == 0
    non_zero_mask = ~zero_mask
    yerr = np.array(
        [
            np.maximum(0, df.loc[non_zero_mask, "received"] - lower_ci[non_zero_mask]),
            np.maximum(0, upper_ci[non_zero_mask] - df.loc[non_zero_mask, "received"]),
        ]
    )

    ax1.errorbar(
        x[non_zero_mask] + 0.2,
        df.loc[non_zero_mask, "received"],
        yerr=yerr,
        fmt="none",
        color=OSRS_COLORS.error_bars,
        alpha=0.6,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Value labels
    for i, v in enumerate(df["received"]):
        x_pos = x[i] + 0.2
        y_pos = -0.5 if v == 0 else v + (upper_ci[i] - df["received"].iloc[i]) + 0.5
        va = "top" if v == 0 else "bottom"
        ax1.text(
            x_pos,
            y_pos,
            f"{v}",
            ha="center",
            va=va,
            color=OSRS_COLORS.text,
            fontweight="bold",
            fontsize=14,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["item_id"].tolist(), rotation=45, ha="right")
    ax1.set_title("Expected vs Actual Drops", pad=20, fontsize=24)
    ax1.legend(facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border)
    ax1.grid(True, alpha=0.2, color=OSRS_COLORS.grid)

    # Search Distribution Plot (ax2)
    ax2.set_facecolor(OSRS_COLORS.panel)
    sns.histplot(
        data=simulated_searches,
        kde=True,
        color=OSRS_COLORS.gold,
        line_kws={"color": OSRS_COLORS.highlight, "alpha": 0.8, "linewidth": 2},
        ax=ax2,
    )
    ax2.axvline(
        float(np.mean(simulated_searches)),
        color=OSRS_COLORS.highlight,
        linestyle="--",
        label="Average",
        alpha=0.8,
        linewidth=2,
    )
    ax2.set_title("Drop Rate Simulation Distribution", pad=20, fontsize=24)
    ax2.set_xlabel("Number of Searches Required")
    ax2.set_ylabel("Frequency")
    ax2.legend(facecolor=OSRS_COLORS.background, edgecolor=OSRS_COLORS.border)
    ax2.grid(True, alpha=0.3, color=OSRS_COLORS.grid)

    # Dryness Analysis Plot (ax3)
    dry_items = df[df["received"] == 0]
    if not dry_items.empty:
        ax3.set_facecolor(OSRS_COLORS.panel)
        plot_data = pd.DataFrame(
            {
                "Item": list(dryness_results.keys()),
                "Times Over Rate": list(dryness_results.values()),
            }
        ).sort_values("Times Over Rate", ascending=True)

        bars = ax3.barh(
            plot_data["Item"],
            plot_data["Times Over Rate"],
            color=OSRS_COLORS.gold,
            alpha=0.9,
        )
        ax3.axvline(
            x=1,
            color="black",
            linestyle="--",
            alpha=0.7,
            label="Expected Rate",
            linewidth=2,
        )

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            x_pos = width - (width * 0.05)
            ax3.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}x",
                ha="right",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=18,
            )

        ax3.set_title("Dry Items Analysis", pad=20, fontsize=24)
        ax3.set_xlabel("Times Over Expected Drop Rate")
        ax3.set_ylabel("Item")

    plt.tight_layout(pad=3.0)
    fig.savefig(
        plots_dir / "combined_analysis.png",
        facecolor=OSRS_COLORS.background,
        bbox_inches="tight",
        dpi=400,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate drop rate statistics")
    parser.add_argument("data_file", type=Path, help="Path to data CSV file")
    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=100,
        help="Number of simulations to run",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--bootstrap-samples",
        "-b",
        type=int,
        default=1000,
        help="Number of bootstrap samples",
    )

    args = parser.parse_args()

    try:
        # Read data from file
        df = read_data(args.data_file)

        # Run simulations
        simulated_searches = simulate_searches_multi(df, args.simulations)
        total_searches = float(np.mean(simulated_searches))
        confidence_interval = np.percentile(simulated_searches, [5, 95])
        dryness = calculate_dryness(df, total_searches)
        dryness_results = dryness if isinstance(dryness, dict) else {}

        # Print results
        print(f"\nEstimated total searches: {total_searches:,.0f}")
        print(
            f"90% confidence interval: [{confidence_interval[0]:,.0f}, {confidence_interval[1]:,.0f}]"
        )
        print("\nDryness analysis for items not yet received:")
        if isinstance(dryness, dict):
            for item_id, times_over in dryness.items():
                item_drop_rate = float(
                    df.loc[df["item_id"] == item_id, "drop_rate"].iloc[0]
                )
                expected_drops = total_searches * item_drop_rate
                prob_zero = (1 - item_drop_rate) ** total_searches
                print(f"{item_id}:")
                print(
                    f"  Expected drops after {total_searches:.0f} searches: {expected_drops:.2f}"
                )
                print("  Actual drops: 0")
                print(f"  Probability of being this dry: {prob_zero:.20f}")
                print(f"  Times over expected rate: {times_over:.2f}x")

        # Plot results with custom output directory
        plot_results(
            df, simulated_searches, total_searches, dryness_results, args.output_dir
        )
        print(f"\nPlots have been saved to the '{args.output_dir}' directory.")

    except Exception as e:
        print(f"Error: {e!s}")
        raise


if __name__ == "__main__":
    main()

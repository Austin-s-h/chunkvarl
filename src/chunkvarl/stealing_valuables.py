import argparse
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt
from tqdm import tqdm

from chunkvarl.plotting import OSRS_COLORS, apply_osrs_style


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
        # Combine equivalent statuettes by summing their drop rates and received counts
        if "blessed_bone_statuette" in df["item_id"].values:
            mask = df["item_id"] == "blessed_bone_statuette"
            combined_rate = df[mask]["drop_rate"].sum()
            combined_received = df[mask]["received"].sum()

            # Remove all statuette entries
            df = df[~mask]

            # Add single combined entry
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "item_id": "blessed_bone_statuette",
                                "drop_rate": combined_rate,
                                "received": combined_received,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        # Validate with Pydantic
        items_data = df.to_dict("records")
        validated_data = ItemCollection(items=[Item(**{str(k): v for k, v in item.items()}) for item in items_data])
        return pd.DataFrame([item.model_dump() for item in validated_data.items])

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file: {file_path}")
    except Exception as e:
        raise ValueError(f"Error validating data: {e!s}")


def simulate_searches(df: pd.DataFrame, seed: int) -> int:
    """Simulate searches including probability of empty searches.
    In Varlamore thieving, most searches yield nothing. When a successful
    roll occurs, one item from the drop table is selected.
    Returns:
        int: Number of searches performed (always >= 1)
    """
    np.random.seed(seed)
    items_collected = {item_id: 0 for item_id in df["item_id"]}
    required_counts = dict(zip(df["item_id"], df["received"]))
    total_searches = 0
    # Calculate total drop rate (sum of all individual rates)
    total_drop_rate = df["drop_rate"].sum()
    if total_drop_rate > 1:
        raise ValueError("Total drop rate cannot exceed 1.0")
    # Probability of getting nothing = 1 - sum of all drop rates
    null_prob = 1 - total_drop_rate
    # Always perform at least one search
    total_searches = 1
    # First search
    if np.random.random() >= null_prob:
        normalized_rates = df["drop_rate"] / total_drop_rate
        item_idx = np.random.choice(len(df), p=normalized_rates)
        item_id = df.iloc[item_idx]["item_id"]
        if items_collected[item_id] < required_counts[item_id]:
            items_collected[item_id] += 1

    # Continue searching if needed
    while any(items_collected[item] < required_counts[item] for item in items_collected):
        total_searches += 1

        if np.random.random() >= null_prob:
            normalized_rates = df["drop_rate"] / total_drop_rate
            item_idx = np.random.choice(len(df), p=normalized_rates)
            item_id = df.iloc[item_idx]["item_id"]
            if items_collected[item_id] < required_counts[item_id]:
                items_collected[item_id] += 1

    return total_searches


def calculate_dryness_for_item(item_id: str, drop_rate: float, num_simulations: int) -> tuple[str, float]:
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


def calculate_dryness(df: pd.DataFrame, total_searches: float) -> str | dict[str, float]:
    """Calculate dryness for items with 0 drops using modified geometric distribution."""
    if df.empty:
        raise ValueError("DataFrame is empty")

    dry_items = df[df["received"] == 0]
    if dry_items.empty:
        return "No items with a value of 0 in your data."

    # Total probability of getting any item
    total_drop_rate = df["drop_rate"].sum()
    null_prob = 1 - total_drop_rate

    results = {}
    for _, row in dry_items.iterrows():
        p = row["drop_rate"]
        # Expected attempts including null searches
        expected_attempts = 1 / p
        # Variance for geometric distribution with null probability
        variance = (1 - p) / (p**2)
        std_dev = np.sqrt(variance)

        # Probability of being this dry (zero drops in total_searches)
        prob_zero = (1 - p) ** total_searches

        # Calculate how many standard deviations from mean
        times_over_rate = (total_searches - expected_attempts) / std_dev
        results[row["item_id"]] = max(times_over_rate, 0)

    return results


def _bootstrap_wrapper(params: tuple[pd.DataFrame, int]) -> NDArray[np.float64]:
    """Wrapper function for bootstrap sampling."""
    df, _ = params
    return np.random.choice(df["received"], size=len(df), replace=True)


def calculate_bootstrap_errors(
    df: pd.DataFrame, num_bootstrap: int = 1000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate bootstrap confidence intervals accounting for null searches."""
    with mp.Pool(mp.cpu_count()) as pool:
        params = [(df, i) for i in range(num_bootstrap)]
        bootstrap_results = list(
            tqdm(
                pool.imap(_bootstrap_wrapper, params),
                total=num_bootstrap,
                desc="Calculating bootstrap intervals",
            )
        )

    # Calculate expected drops including null probability
    total_drop_rate = df["drop_rate"].sum()
    bootstrap_array = np.array(
        [[n * (p / total_drop_rate) for p in df["drop_rate"]] for n in [sample.sum() for sample in bootstrap_results]]
    )

    lower_ci = np.percentile(bootstrap_array, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_array, 97.5, axis=0)

    return lower_ci, upper_ci


def calculate_current_dryness_percentile(simulated_searches: list[int], total_searches: float) -> float:
    """Calculate what percentile of dryness the current total_searches represents."""
    return sum(1 for x in simulated_searches if x <= total_searches) / len(simulated_searches) * 100

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

    # Create figure with GridSpec with adjusted margins
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.05, wspace=0.05)

    # Increase default font sizes
    plt.rc("font", size=14)  # controls default text size
    plt.rc("axes", titlesize=24)  # fontsize of the title
    plt.rc("axes", labelsize=16)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=18)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=14)  # fontsize of the y tick labels
    plt.rc("legend", fontsize=14)  # fontsize of the legend

    # Create subplots
    ax1 = fig.add_subplot(gs[0, :])  # Expected vs Actual (spans both columns)
    ax2 = fig.add_subplot(gs[1, 0])  # Search Distribution
    ax3 = fig.add_subplot(gs[1, 1])  # Dryness Analysis

    # Calculate dryness colors
    def get_dryness_color(actual: float, expected: float, lower: float, upper: float) -> tuple[float, float, float]:
        if actual == 0:
            # Red for dry items, darker red for more significantly dry
            dryness = (expected - 0) / expected
            return (min(1.0, 0.4 + dryness * 0.6), 0.0, 0.0)  # Varying shades of red
        elif actual < expected:
            # Yellow to red gradient for below expected
            ratio = actual / expected
            return (1.0, ratio, 0.0)  # Yellow to red
        else:
            # Green for above expected, brighter green for luckier
            luck = min((actual - expected) / (upper - expected) if upper > expected else 0.0, 1.0)
            return (0.0, min(0.8 + luck * 0.2, 1.0), 0.0)  # Varying shades of green

    # Sort items by rarity (drop rate)
    df = df.sort_values(by="drop_rate", ascending=True).reset_index(drop=True)

    # Expected vs Actual Plot (ax1)
    ax1.set_facecolor(OSRS_COLORS.panel)
    x = np.arange(len(df))
    expected_drops = total_searches * df["drop_rate"]
    lower_ci, upper_ci = calculate_bootstrap_errors(df)

    param_text = (
        f"Simulation Parameters\n"
        f"Total Simulations: {len(simulated_searches):,}\n"
        f"Total Drop Rate: {df['drop_rate'].sum():.4f}\n"
        f"Avg. Searches: {total_searches:,.0f}\n"
        f"Items Tracked: {len(df)}"
    )

    right_align = 0.02
    initial_top = 0.98
    box_padding = -0.20

    ax1.text(
        right_align,
        initial_top,
        param_text,
        transform=ax1.transAxes,
        fontsize=10,
        color=OSRS_COLORS.text,
        bbox=dict(facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border, alpha=0.8, boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="left",
    )

    bootstrap_stats = "Bootstrap Statistics (95% CI):\n" + "\n".join(
        f"{item_id}:\n  {lower:.1f} - {upper:.1f}" for item_id, lower, upper in zip(df["item_id"], lower_ci, upper_ci)
    )

    ax1.text(
        right_align,
        initial_top + box_padding,
        bootstrap_stats,
        transform=ax1.transAxes,
        fontsize=10,
        color=OSRS_COLORS.text,
        bbox=dict(facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border, alpha=0.8, boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="left",
    )

    ax1.legend(
        bbox_to_anchor=(0.02, 0.02),
        loc="lower left",
        facecolor=OSRS_COLORS.panel,
        edgecolor=OSRS_COLORS.border,
        framealpha=0.8,
    )

    # Plot expected bars
    ax1.bar(x - 0.2, expected_drops, width=0.4, color=OSRS_COLORS.gold, alpha=0.9, label="Expected")
    # Plot actual bars
    ax1.bar(x + 0.2, df["received"], width=0.4, color=OSRS_COLORS.blue, alpha=0.9, label="Actual")

    # Error bars logic for non-zero values
    zero_mask = df["received"] == 0
    non_zero_mask = ~zero_mask
    yerr = np.array(
        [
            np.maximum(0, df.loc[non_zero_mask, "received"] - lower_ci[non_zero_mask]),
            np.maximum(0, upper_ci[non_zero_mask] - df.loc[non_zero_mask, "received"]),
        ]
    )

    # Update error bar style
    ax1.errorbar(
        x[non_zero_mask] + 0.2,
        df.loc[non_zero_mask, "received"],
        yerr=yerr,
        fmt="none",
        color=OSRS_COLORS.error_bars,
        alpha=0.8,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Improved value label positioning
    max_height = max(max(expected_drops), max(df["received"]))
    label_padding = max_height * 0.02  # 2% of max height for padding

    # Value labels with improved positioning
    for i, (actual, expected, upper_ci_val) in enumerate(zip(df["received"], expected_drops, upper_ci)):
        # Expected value label
        ax1.text(
            x[i] - 0.2,
            expected + label_padding,
            f"{expected:.0f}",
            ha="center",
            va="bottom",
            color=OSRS_COLORS.text,
            fontweight="bold",
            fontsize=14,
            bbox=dict(
                facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border, alpha=0.7, pad=2, boxstyle="round,pad=0.3"
            ),
        )

        # Actual value label
        if actual == 0:
            y_pos = 0.02
            va = "bottom"
        else:
            y_pos = actual + label_padding
            # Add extra padding if close to expected value
            if abs(actual - expected) < max_height * 0.1:
                y_pos += label_padding
            va = "bottom"

        ax1.text(
            x[i] + 0.2,
            y_pos,
            f"{actual}",
            ha="center",
            va=va,
            color=OSRS_COLORS.text,
            fontweight="bold",
            fontsize=14,
            bbox=dict(
                facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border, alpha=0.7, pad=2, boxstyle="round,pad=0.3"
            ),
        )

    # Add colored rectangles and drop rate annotations
    for i, (drop_rate, item_id) in enumerate(zip(df["drop_rate"], df["item_id"])):
        # Get rarity color based on drop rate
        rarity_color = OSRS_COLORS.get_rarity_tier(drop_rate).color

        # Add background rectangle
        rect = plt.Rectangle(
            (i - 0.4, -max_height * 0.08),
            0.8,
            max_height * 0.06,
            facecolor=rarity_color,
            alpha=0.3,
            edgecolor=OSRS_COLORS.border,
            linewidth=1,
        )
        ax1.add_patch(rect)

        # Add the drop rate text
        ax1.text(
            i,
            -max_height * 0.05,
            f"1/{1 / drop_rate:.0f}",
            ha="center",
            va="center",
            color=OSRS_COLORS.text,
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="none", edgecolor="none", pad=2),
        )

    # Adjust y-axis limit to accommodate colored rectangles
    ax1.set_ylim(bottom=-max_height * 0.10 if any(df["received"] == 0) else -max_height * 0.1, top=max_height * 1.15)

    # Modify the x-axis label settings for better alignment
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        df["item_id"].tolist(),
        rotation=45,
        ha="right",
        va="top",  # Add vertical alignment
    )
    # Add padding to prevent label overlap
    ax1.tick_params(axis="x", pad=10)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["item_id"].tolist(), rotation=45, ha="right")
    ax1.set_title("Expected vs Actual Drops", pad=20, fontsize=24)
    ax1.grid(True, alpha=0.2, color=OSRS_COLORS.grid)

    # Search Distribution Plot (ax2) - Modified
    ax2.set_facecolor(OSRS_COLORS.panel)
    
    # Calculate current dryness percentile
    current_percentile = calculate_current_dryness_percentile(simulated_searches, total_searches)
    
    # Plot histogram with KDE
    sns.histplot(
        data=simulated_searches,
        kde=True,
        color=OSRS_COLORS.gold,
        line_kws={"color": OSRS_COLORS.highlight, "alpha": 0.8, "linewidth": 3},
        ax=ax2,
    )

    # Add mean and current position lines
    mean_val = float(np.mean(simulated_searches))
    ax2.axvline(
        mean_val,
        color=OSRS_COLORS.dashed_lines,
        linestyle="--",
        label="Average",
        alpha=0.8,
        linewidth=3,
    )
    
    # Add current position line
    ax2.axvline(
        total_searches,
        color='red',
        linestyle='-',
        label=f'Current ({current_percentile:.1f}%ile)',
        alpha=0.8,
        linewidth=3,
    )

    # Update stats text to include current position
    mean_searches = float(np.mean(simulated_searches))
    std_searches = float(np.std(simulated_searches))
    stats_text = (
        f"Distribution Statistics:\n"
        f"Mean: {mean_searches:,.0f}\n"
        f"Current: {total_searches:,.0f}\n"
        f"Percentile: {current_percentile:.1f}%\n"
        f"Std Dev: {std_searches:,.0f}\n"
        f"5th %ile: {np.percentile(simulated_searches, 5):,.0f}\n"
        f"95th %ile: {np.percentile(simulated_searches, 95):,.0f}"
    )

    ax2.text(
        0.90,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        color=OSRS_COLORS.text,
        bbox=dict(facecolor=OSRS_COLORS.panel, edgecolor=OSRS_COLORS.border, alpha=0.8, boxstyle="round,pad=0.5"),
        verticalalignment="top",
        horizontalalignment="right",
    )

    # Dryness Analysis Plot (ax3)
    dry_items = df[df["received"] == 0]
    if not dry_items.empty:
        ax3.set_facecolor(OSRS_COLORS.panel)
        plot_data = pd.DataFrame(
            {
                "Item": list(dryness_results.keys()),
                "Times Over Rate": list(dryness_results.values()),
                "Drop Rate": [df.loc[df["item_id"] == item, "drop_rate"].iloc[0] for item in dryness_results],
            }
        ).sort_values("Times Over Rate", ascending=True)

        # Create bars with rarity-based colors
        bars = ax3.barh(
            plot_data["Item"],
            plot_data["Times Over Rate"],
            color=[OSRS_COLORS.get_rarity_color(rate) for rate in plot_data["Drop Rate"]],
            alpha=0.9,
            edgecolor=OSRS_COLORS.border,
            linewidth=1,
        )

        # Add reference line for expected rate with updated legend
        ax3.axvline(
            x=1,
            color=OSRS_COLORS.dashed_lines,
            linestyle="--",
            alpha=0.8,
            label="Expected Drop Rate",
            linewidth=3,
        )

        # Add value labels with improved styling
        for bar in bars:
            width = bar.get_width()
            x_pos = width + (width * 0.02)  # Slightly offset from end of bar
            ax3.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}x",
                ha="left",
                va="center",
                color=OSRS_COLORS.text,
                fontweight="bold",
                fontsize=14,
                bbox=dict(
                    facecolor=OSRS_COLORS.panel,
                    edgecolor=OSRS_COLORS.border,
                    alpha=0.7,
                    pad=2,
                    boxstyle="round,pad=0.3",
                ),
            )

        # Add colored rectangles and drop rates to y-axis labels
        yticks = ax3.get_yticks()
        ylabels = [plot_data["Item"].iloc[int(i)] for i in range(len(yticks))]
        new_labels = []

        # Add background rectangles for each item
        for i, (item, drop_rate) in enumerate(zip(plot_data["Item"], plot_data["Drop Rate"])):
            # Get rarity color based on drop rate
            rarity_color = OSRS_COLORS.get_rarity_tier(drop_rate).color
            # Add background rectangle
            rect = plt.Rectangle(
                (ax3.get_xlim()[0], i - 0.4),  # Start from left edge
                ax3.get_xlim()[1] - ax3.get_xlim()[0],  # Span full width
                0.8,
                facecolor=rarity_color,
                alpha=0.1,
                edgecolor=OSRS_COLORS.border,
                linewidth=1,
            )
            ax3.add_patch(rect)

            # Calculate probability of zero drops for this item
            prob_zero = (1 - drop_rate) ** total_searches
            # Create label with drop rate
            new_labels.append(f"{item}\n(1/{1/drop_rate:.0f}) p₀={prob_zero:.1e}")

        # Set y-axis ticks and labels
        ax3.set_yticks(range(len(plot_data)))
        ax3.set_yticklabels(new_labels)

        # Style the plot
        ax3.set_title("Dry Items Analysis", pad=20, fontsize=24)
        ax3.set_xlabel("Times Over Expected Drop Rate", fontsize=16)
        ax3.set_ylabel("")  # Remove y-axis label since we have item names
        ax3.grid(True, alpha=0.3, color=OSRS_COLORS.grid, axis="x")
        ax3.margins(x=0.05)  # Reduce right margin since we no longer need space for p₀

    # Adjust overall figure margins
    fig.savefig(
        plots_dir / "combined_analysis.png",
        facecolor=OSRS_COLORS.background,
        bbox_inches="tight",
        dpi=400,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate stealing valuables from varlamore thieving.")
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to stolen valuables in CSV format. Must contain 'item_id', 'drop_rate', and 'received' columns.",
    )
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

        # Print enhanced results
        print(f"\nEstimated total searches: {total_searches:,.0f}")
        print(f"90% confidence interval: [{confidence_interval[0]:,.0f}, {confidence_interval[1]:,.0f}]")
        current_percentile = calculate_current_dryness_percentile(simulated_searches, total_searches)
        print(f"Current dryness percentile: {current_percentile:.1f}%")
        print("\nDryness analysis for items not yet received:")
        if isinstance(dryness, dict):
            for item_id, times_over in dryness.items():
                item_drop_rate = float(df.loc[df["item_id"] == item_id, "drop_rate"].iloc[0])
                expected_drops = total_searches * item_drop_rate
                prob_zero = (1 - item_drop_rate) ** total_searches
                print(f"{item_id}:")
                print(f"  Expected drops after {total_searches:.0f} searches: {expected_drops:.2f}")
                print("  Actual drops: 0")
                print(f"  Probability of being this dry: {prob_zero:.10f}")
                print(f"  Times over expected rate: {times_over:.2f}x")
                print(f"  Current search count is at {current_percentile:.1f}%ile of simulations")

        # Plot results with custom output directory
        plot_results(df, simulated_searches, total_searches, dryness_results, args.output_dir)
        print(f"\nPlots have been saved to the '{args.output_dir}' directory.")

    except Exception as e:
        print(f"Error: {e!s}")
        raise


if __name__ == "__main__":
    main()

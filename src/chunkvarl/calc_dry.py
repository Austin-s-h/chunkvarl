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
        validated_data = ItemCollection(items=[Item(**{str(k): v for k, v in item.items()}) for item in items_data])

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
    while any(items_collected[item_id] < received for item_id, received in zip(df["item_id"], df["received"])):
        for item_id, drop_rate in zip(df["item_id"], df["drop_rate"]):
            if np.random.random() < drop_rate:
                items_collected[item_id] += 1
                break
        searches += 1
    return searches


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
    """Generate and save analysis plots."""
    plots_dir = Path(output_dir)
    try:
        plots_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        raise PermissionError(f"Cannot create plots directory at {plots_dir}")

    # Plot 1: Distribution of simulated searches
    plt.figure(figsize=(12, 6))
    sns.histplot(simulated_searches, kde=True)
    plt.title("Distribution of Simulated Searches")
    plt.xlabel("Number of Searches")
    plt.axvline(float(np.mean(simulated_searches)), color="r", linestyle="--", label="Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "search_distribution.png", dpi=300)
    plt.close()

    # Plot 2: Expected vs Actual drops with error bars
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    x = np.arange(len(df))
    expected_drops = total_searches * df["drop_rate"]
    lower_ci, upper_ci = calculate_bootstrap_errors(df)

    # Plot bars first
    plt.bar(
        x - 0.2,
        expected_drops,
        width=0.4,
        label="Expected",
        color=sns.color_palette("husl")[0],
        alpha=0.7,
    )
    plt.bar(
        x + 0.2,
        df["received"],
        width=0.4,
        label="Actual",
        color=sns.color_palette("husl")[1],
        alpha=0.7,
    )

    # Calculate error bar heights (ensure they're non-negative)
    yerr = np.array(
        [
            np.maximum(0, df["received"] - lower_ci),  # lower errors
            np.maximum(0, upper_ci - df["received"]),  # upper errors
        ]
    )

    # Add error bars to the actual values
    plt.errorbar(
        x + 0.2,  # Changed from x - 0.2 to x + 0.2 to align with "Actual" bars
        df["received"],  # Changed from expected_drops to df["received"]
        yerr=yerr,
        fmt="none",
        color="black",
        alpha=0.3,
        capsize=5,
    )

    # Customize appearance
    plt.xticks(x, df["item_id"].tolist(), rotation=45, ha="right")
    plt.title("Expected vs Actual Drops\nwith 95% Confidence Intervals", pad=20, fontsize=14)
    plt.xlabel("Items", fontsize=12, labelpad=10)
    plt.ylabel("Number of Drops", fontsize=12, labelpad=10)

    # Add legend with custom styling
    legend = plt.legend(frameon=True, facecolor="white", framealpha=1)
    frame = legend.get_frame()
    frame.set_edgecolor("black")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(plots_dir / "expected_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Dryness Analysis
    dry_items = df[df["received"] == 0]
    if not dry_items.empty:
        plt.figure(figsize=(10, 6))

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(
            {
                "Item": list(dryness_results.keys()),
                "Times Over Rate": list(dryness_results.values()),
            }
        )

        # Sort by dryness
        plot_data = plot_data.sort_values("Times Over Rate", ascending=True)

        # Create horizontal bar plot
        ax = sns.barplot(data=plot_data, y="Item", x="Times Over Rate", palette="YlOrRd", orient="h")

        # Add vertical line for expected rate
        plt.axvline(x=1, color="blue", linestyle="--", alpha=0.5, label="Expected Rate")

        # Add value labels on the bars
        for i, v in enumerate(plot_data["Times Over Rate"]):
            ax.text(v, i, f"{v:.1f}x", va="center", fontweight="bold")

        plt.title("Dryness Analysis\n(Higher = More Unlucky)", pad=20)
        plt.xlabel("Times Over Expected Drop Rate")
        plt.ylabel("Item")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            plots_dir / "dryness_analysis.png",
            dpi=300,
            bbox_inches="tight",
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
        print(f"90% confidence interval: [{confidence_interval[0]:,.0f}, {confidence_interval[1]:,.0f}]")
        print("\nDryness analysis for items not yet received:")
        if isinstance(dryness, dict):
            for item_id, times_over in dryness.items():
                item_drop_rate = float(df.loc[df["item_id"] == item_id, "drop_rate"].iloc[0])
                expected_drops = total_searches * item_drop_rate
                prob_zero = (1 - item_drop_rate) ** total_searches
                print(f"{item_id}:")
                print(f"  Expected drops after {total_searches:.0f} searches: {expected_drops:.2f}")
                print("  Actual drops: 0")
                print(f"  Probability of being this dry: {prob_zero:.20f}")
                print(f"  Times over expected rate: {times_over:.2f}x")

        # Plot results with custom output directory
        plot_results(df, simulated_searches, total_searches, dryness_results, args.output_dir)
        print(f"\nPlots have been saved to the '{args.output_dir}' directory.")

    except Exception as e:
        print(f"Error: {e!s}")
        raise


if __name__ == "__main__":
    main()

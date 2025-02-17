import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Placeholder data
data = {
    "item_id": [
        "blessed_bone_statuette",
        "gold_necklace",
        "gold_amulet",
        "gold_ring",
        "emerald_amulet",
        "sapphire_amulet",
        "emerald_necklace",
        "sapphire_necklace",
        "ruby_amulet",
        "ruby_necklace",
        "diamond_amulet",
        "ruby_ring",
        "emerald_ring",
        "sapphire_ring",
        "diamond_ring",
        "diamond_necklace",
    ],
    "drop_rate": [
        1 / 520.8,
        1 / 625,
        1 / 625,
        1 / 625,
        1 / 625,
        1 / 625,
        1 / 694.4,
        1 / 694.4,
        1 / 1250,
        1 / 1250,
        1 / 2083,
        1 / 3125,
        1 / 3125,
        1 / 3125,
        1 / 6250,
        1 / 6250,
    ],
    "received": [
        74,
        27,
        13,
        18,
        31,
        22,
        20,
        15,
        14,
        12,
        9,
        5,
        0,
        8,
        1,
        0,
    ],
}

df = pd.DataFrame(data)


def simulate_searches(df: pd.DataFrame, seed: int) -> int:
    np.random.seed(seed)
    searches: int = 0
    items_collected: dict[str, int] = {item_id: 0 for item_id in df["item_id"]}
    while any(items_collected[item_id] < received for item_id, received in zip(df["item_id"], df["received"])):  # type: ignore
        searches += 1
        for item_id, drop_rate in zip(df["item_id"], df["drop_rate"]):
            if np.random.random() < drop_rate:
                items_collected[item_id] += 1
                break  # Stop after collecting one item in each search
    return searches


def calculate_dryness_for_item(item_id: str, drop_rate: float, num_simulations: int) -> tuple[str, float]:
    searches_list: list[int] = []
    for _ in range(num_simulations):
        searches = 0
        found = False
        while not found:
            searches += 1
            if np.random.random() < drop_rate:
                found = True
        searches_list.append(searches)
    return item_id, np.mean(searches_list)


def simulate_searches_multi(df: pd.DataFrame, num_simulations: int) -> list[int]:
    with mp.Pool(mp.cpu_count()) as pool:
        seeds = np.random.randint(0, 1e8, size=num_simulations)
        results = pool.starmap(simulate_searches, [(df, seed) for seed in seeds])
    return results


def calculate_dryness(df: pd.DataFrame, total_searches: float) -> str | dict[str, float]:
    """Calculate dryness for items with 0 drops using the estimated total searches."""
    dry_items = df[df["received"] == 0]
    if dry_items.empty:
        return "No items with a value of 0 in your data."

    results = {}
    for _, row in dry_items.iterrows():
        # Expected number of drops after total_searches attempts
        expected_drops = total_searches * row["drop_rate"]
        # Probability of getting 0 drops in total_searches attempts
        prob_zero = (1 - row["drop_rate"]) ** total_searches
        # Times over expected rate (using probability)
        times_over_rate = -np.log(prob_zero) if prob_zero > 0 else float("inf")
        results[row["item_id"]] = times_over_rate

    return results


def calculate_bootstrap_errors(df: pd.DataFrame, num_bootstrap: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Calculate bootstrap confidence intervals for expected drops."""
    bootstrap_results = []
    n_items = len(df)

    for _ in range(num_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(df["received"], size=len(df), replace=True)
        total_searches = bootstrap_sample.sum() / np.sum(df["drop_rate"])
        expected_drops = total_searches * df["drop_rate"]
        bootstrap_results.append(expected_drops)

    bootstrap_array = np.array(bootstrap_results)
    lower_ci = np.percentile(bootstrap_array, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_array, 97.5, axis=0)

    return lower_ci, upper_ci


def plot_results(
    df: pd.DataFrame, simulated_searches: list[int], total_searches: float, dryness_results: dict[str, float]
) -> None:
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Distribution of simulated searches
    plt.figure(figsize=(12, 6))
    sns.histplot(simulated_searches, kde=True)
    plt.title("Distribution of Simulated Searches")
    plt.xlabel("Number of Searches")
    plt.axvline(np.mean(simulated_searches), color="r", linestyle="--", label="Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "search_distribution.png"), dpi=300)
    plt.close()

    # Plot 2: Expected vs Actual drops with error bars
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    x = np.arange(len(df))
    expected_drops = total_searches * df["drop_rate"]
    lower_ci, upper_ci = calculate_bootstrap_errors(df)

    # Plot bars first
    plt.bar(x - 0.2, expected_drops, width=0.4, label="Expected", color=sns.color_palette("husl")[0], alpha=0.7)
    plt.bar(x + 0.2, df["received"], width=0.4, label="Actual", color=sns.color_palette("husl")[1], alpha=0.7)

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
    plt.xticks(x, df["item_id"], rotation=45, ha="right")
    plt.title("Expected vs Actual Drops\nwith 95% Confidence Intervals", pad=20, fontsize=14)
    plt.xlabel("Items", fontsize=12, labelpad=10)
    plt.ylabel("Number of Drops", fontsize=12, labelpad=10)

    # Add legend with custom styling
    legend = plt.legend(frameon=True, facecolor="white", framealpha=1)
    frame = legend.get_frame()
    frame.set_edgecolor("black")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "expected_vs_actual.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Dryness Analysis
    dry_items = df[df["received"] == 0]
    if not dry_items.empty:
        plt.figure(figsize=(10, 6))

        # Create DataFrame for plotting
        plot_data = pd.DataFrame(
            {"Item": list(dryness_results.keys()), "Times Over Rate": list(dryness_results.values())}
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
        plt.savefig(os.path.join(plots_dir, "dryness_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    num_simulations: int = 100

    # Run simulations
    simulated_searches = simulate_searches_multi(df, num_simulations)
    total_searches = np.mean(simulated_searches)  # Use mean as estimate of total searches
    confidence_interval = np.percentile(simulated_searches, [5, 95])
    dryness_results = calculate_dryness(df, total_searches)

    # Print results
    print(f"Estimated total searches: {total_searches:.0f}")
    print(f"90% confidence interval: [{confidence_interval[0]:.0f}, {confidence_interval[1]:.0f}]")
    print("\nDryness analysis for items not yet received:")
    for item_id, times_over in dryness_results.items():
        item_drop_rate = float(df.loc[df["item_id"] == item_id, "drop_rate"].iloc[0])
        expected_drops = total_searches * item_drop_rate
        prob_zero = (1 - item_drop_rate) ** total_searches
        print(f"{item_id}:")
        print(f"  Expected drops after {total_searches:.0f} searches: {expected_drops:.2f}")
        print("  Actual drops: 0")
        print(f"  Probability of being this dry: {prob_zero:.20f}")
        print(f"  Times over expected rate: {times_over:.2f}x")

    # Plot results
    plot_results(df, simulated_searches, total_searches, dryness_results)
    print("\nPlots have been saved to the 'plots' directory.")


if __name__ == "__main__":
    main()

import pandas as pd
from scipy.stats import norm
from utils.visualization import plot_uncertainty_map


def main():
    df = pd.read_csv("outputs/logs/evaluation_predictions.csv")
    z = norm.ppf(0.95)  # 90% interval width: 2 * z_0.95 * sigma
    df["interval_width_90"] = 2 * z * df["sigma"]
    plot_uncertainty_map(df["longitude"], df["latitude"], df["interval_width_90"], "outputs/figures/uncertainty_map.png")
    df.to_csv("outputs/logs/uncertainty_predictions.csv", index=False)


if __name__ == "__main__":
    main()

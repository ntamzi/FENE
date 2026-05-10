import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from models.fene import FENE
from utils.preprocessing import DataProcessor
from utils.dataset import FishDataset
from utils.metrics import rmse, r2
from utils.visualization import plot_predictions


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cfg["data"]["csv_path"])
    processor = DataProcessor.load("outputs/checkpoints/processor.joblib")
    X, species, coords, key = processor.transform(df, target_col=cfg["data"]["target_col"], species_col=cfg["data"]["species_col"])
    y = df[cfg["data"]["target_col"]].values

    dataset = FishDataset(X, species, coords, key, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = FENE(
        input_dim=X.shape[1],
        num_species=len(processor.label_encoder.classes_),
        embedding_dim=cfg["model"]["species_embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        key_feature_dim=cfg["model"]["key_feature_dim"],
        num_experts=cfg["model"]["num_experts"],
        fourier_bands=cfg["model"]["fourier_bands"],
        siren_hidden=cfg["model"]["siren_hidden"],
    ).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
    model.eval()

    preds, sigmas, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, sigma = model(batch["env"].to(device), batch["species"].to(device), batch["coords"].to(device), batch["key"].to(device))
            preds.extend(pred.cpu().numpy())
            sigmas.extend(sigma.cpu().numpy())
            targets.extend(batch["target"].numpy())

    print(f"RMSE: {rmse(targets, preds):.4f}")
    print(f"R2:   {r2(targets, preds):.4f}")
    plot_predictions(targets, preds, "outputs/figures/prediction_scatter.png")

    out = df.copy()
    out["prediction"] = preds
    out["sigma"] = sigmas
    out.to_csv("outputs/logs/evaluation_predictions.csv", index=False)


if __name__ == "__main__":
    main()

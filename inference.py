import argparse
import yaml
import torch
import pandas as pd

from models.fene import FENE
from utils.preprocessing import DataProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file containing samples for prediction")
    parser.add_argument("--output", default="outputs/logs/inference_predictions.csv")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.input)
    processor = DataProcessor.load("outputs/checkpoints/processor.joblib")
    X, species, coords, key = processor.transform(df, target_col=None, species_col=cfg["data"]["species_col"])

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

    with torch.no_grad():
        pred, sigma = model(
            torch.tensor(X).float().to(device),
            torch.tensor(species).long().to(device),
            torch.tensor(coords).float().to(device),
            torch.tensor(key).float().to(device),
        )

    df["prediction"] = pred.cpu().numpy()
    df["sigma"] = sigma.cpu().numpy()
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

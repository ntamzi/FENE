import yaml
import shap
import torch
import pandas as pd
import matplotlib.pyplot as plt

from models.fene import FENE
from utils.preprocessing import DataProcessor


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, species, coords, key):
        super().__init__()
        self.model = model
        self.species = species
        self.coords = coords
        self.key = key

    def forward(self, env):
        n = env.shape[0]
        pred, _ = self.model(env, self.species[:n], self.coords[:n], self.key[:n])
        return pred.unsqueeze(1)


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cfg["data"]["csv_path"]).sample(min(500, len(pd.read_csv(cfg["data"]["csv_path"]))), random_state=42)
    processor = DataProcessor.load("outputs/checkpoints/processor.joblib")
    X, species, coords, key = processor.transform(df, target_col=cfg["data"]["target_col"], species_col=cfg["data"]["species_col"])

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

    X_t = torch.tensor(X).float().to(device)
    sp_t = torch.tensor(species).long().to(device)
    c_t = torch.tensor(coords).float().to(device)
    k_t = torch.tensor(key).float().to(device)

    wrapper = ModelWrapper(model, sp_t, c_t, k_t).to(device)
    background = X_t[:100]
    test = X_t[:300]
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(test)

    plt.figure()
    shap.summary_plot(shap_values[0], test.cpu().numpy(), feature_names=processor.feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_summary.png", dpi=300)
    plt.close()
    print("Saved SHAP summary to outputs/shap/shap_summary.png")


if __name__ == "__main__":
    main()

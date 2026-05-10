import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.fene import FENE
from models.losses import FocalHuberLoss
from utils.preprocessing import DataProcessor
from utils.dataset import FishDataset
from utils.metrics import rmse, r2
from utils.seed import set_seed


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cfg["data"]["csv_path"])

    processor = DataProcessor(
        longitude_col=cfg["data"]["longitude_col"],
        latitude_col=cfg["data"]["latitude_col"],
    )
    X, species, coords, key_features, y, feature_names = processor.fit_transform(
        df,
        target_col=cfg["data"]["target_col"],
        species_col=cfg["data"]["species_col"],
        top_k_key_features=cfg["data"]["top_k_key_features"],
    )
    processor.save("outputs/checkpoints/processor.joblib")

    X_train, X_test, s_train, s_test, c_train, c_test, k_train, k_test, y_train, y_test = train_test_split(
        X, species, coords, key_features, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["seed"],
        stratify=species,
    )

    train_dataset = FishDataset(X_train, s_train, c_train, k_train, y_train)
    test_dataset = FishDataset(X_test, s_test, c_test, k_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = FENE(
        input_dim=X.shape[1],
        num_species=len(np.unique(species)),
        embedding_dim=cfg["model"]["species_embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        key_feature_dim=cfg["model"]["key_feature_dim"],
        num_experts=cfg["model"]["num_experts"],
        fourier_bands=cfg["model"]["fourier_bands"],
        siren_hidden=cfg["model"]["siren_hidden"],
    ).to(device)

    criterion = FocalHuberLoss(delta=cfg["loss"]["huber_delta"], alpha=cfg["loss"]["focal_alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_rmse = float("inf")
    patience_counter = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            env = batch["env"].to(device)
            sp = batch["species"].to(device)
            coord = batch["coords"].to(device)
            key = batch["key"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            pred, sigma = model(env, sp, coord, key)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                pred, _ = model(
                    batch["env"].to(device),
                    batch["species"].to(device),
                    batch["coords"].to(device),
                    batch["key"].to(device),
                )
                preds.extend(pred.cpu().numpy())
                targets.extend(batch["target"].numpy())

        epoch_rmse = rmse(targets, preds)
        epoch_r2 = r2(targets, preds)
        print(f"Epoch {epoch+1:03d} | Loss={total_loss:.4f} | RMSE={epoch_rmse:.4f} | R2={epoch_r2:.4f}")

        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            patience_counter = 0
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()

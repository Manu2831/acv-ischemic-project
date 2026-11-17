import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,recall_score, precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

# 1. Cargar datos

df_train = pd.read_csv("../datos/df_train_acv.csv")
df_val   = pd.read_csv("../datos/df_val_acv.csv")
df_test  = pd.read_csv("../datos/df_test_acv.csv")

#  Unimos TODO para KFold 
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)


# 2. Variables

features = ["Edad", "NIHSS ingreso", "Fibrilación Auricular", "Hipertensión"]
target = "mRS 90 días"


# 3. Limpieza de target

df_all[target] = pd.to_numeric(df_all[target], errors="coerce")
df_all = df_all[~df_all[target].isna()].copy()

#  BINARIZACIÓN
df_all["mRS_bin"] = (df_all[target] > 2).astype(int)

# 4. Preparar X e y

X = df_all[features].values
y = df_all["mRS_bin"].values
num_pos = np.sum(y == 1)
num_neg = np.sum(y == 0)

# 5. Modelo

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim=16, num_heads=2, num_layers=2):
        super().__init__()

        # embedding por feature
        self.feature_embed = nn.Linear(1, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=32,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: (batch, num_features)
        x = x.unsqueeze(-1)             # (batch, num_features, 1)
        x = self.feature_embed(x)       # (batch, num_features, emb_dim)
        x = self.transformer(x)         # (batch, num_features, emb_dim)
        # pooling
        embedding = x.mean(dim=1)       # (batch, emb_dim)
        out = self.head(embedding)
        return out, embedding
# 6. K-Fold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

    model = TabularTransformer(input_dim=X.shape[1],emb_dim=16,num_heads=2,num_layers=2)
    
    #for name, param in model.transformer.named_parameters():
     #   if "layers.0" in name:
      #      param.requires_grad = False
            
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # ENTRENAMIENTO

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()


    #  VALIDACIÓN (ARREGLADO)

    model.eval()
    with torch.no_grad():
        val_outputs, val_embeddings = model(X_val_tensor)

        probs = torch.sigmoid(val_outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)

        y_true = y_val_tensor.numpy().flatten()

        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds)
        recall = recall_score(y_true, preds)
        precision = precision_score(y_true, preds)

        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = 0.0

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Recall: {recall:.4f}")

    #  GUARDAR RESULTADOS 
    fold_results.append({
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "recall": recall,
        "precision": precision
    })

# RESULTADOS FINALES
print("\n RESULTADOS PROMEDIO:")

print("Accuracy:", np.mean([f["acc"] for f in fold_results]))
print("F1:", np.mean([f["f1"] for f in fold_results]))
print("AUC:", np.mean([f["auc"] for f in fold_results]))
print("Recall:", np.mean([f["recall"] for f in fold_results]))
print("Precision:", np.mean([f["precision"] for f in fold_results]))



"""
bootstrap.py
------------
Used by the Streamlit app so the live demo works even with no pre-trained
model committed to the repo. If checkpoints/hydro_tgt.pt and dataset/graph.pt
don't exist yet, this generates a small synthetic wallet-transaction graph
and trains HydroTGT on it for a handful of epochs — enough for a convincing
live demo, not a research-grade model.

For a real model, train properly (see notebooks/Hydro-TGT.ipynb) and commit
the resulting checkpoints/hydro_tgt.pt + dataset/graph.pt instead — bootstrap
will then be skipped automatically.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.join(ROOT, "models"))
sys.path.append(os.path.join(ROOT, "preprocessing"))

from classifier import HydroTGT  # noqa: E402
from clean_data import generate_synthetic, clean_and_validate  # noqa: E402
from graph_builder import build_graph  # noqa: E402

GRAPH_PATH = os.path.join(ROOT, "dataset", "graph.pt")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "hydro_tgt.pt")
NODES_PATH = os.path.join(ROOT, "dataset", "clean", "nodes.csv")


def ensure_demo_artifacts(n_wallets: int = 300, n_tx: int = 3000, epochs: int = 25):
    """Creates graph.pt + checkpoint if they don't already exist. Returns (graph_path, checkpoint_path)."""
    if os.path.exists(GRAPH_PATH) and os.path.exists(CHECKPOINT_PATH):
        return GRAPH_PATH, CHECKPOINT_PATH

    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(NODES_PATH), exist_ok=True)

    nodes, transactions, labels = generate_synthetic(n_wallets, n_tx)
    nodes, transactions, labels = clean_and_validate(nodes, transactions, labels)
    nodes.to_csv(NODES_PATH, index=False)  # needed by the app for wallet_id lookups

    data = build_graph(nodes, transactions, labels, seq_len=16)
    torch.save(data, GRAPH_PATH)

    model = HydroTGT(static_in_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_y = data.y[data.train_mask]
    n_pos = (train_y == 1).sum().item()
    n_neg = (train_y == 0).sum().item()
    class_weights = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.temporal_x)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    return GRAPH_PATH, CHECKPOINT_PATH

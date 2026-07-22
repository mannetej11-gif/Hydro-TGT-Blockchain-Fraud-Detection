"""
streamlit_app.py
-----------------
Self-contained Hydro-TGT demo — everything (model, data generation, training,
inference) lives in this single file so deployment can't break on folder/
import-path mismatches. Set this as the Main file path in Streamlit Cloud
(Manage app → Settings → General → Main file path: streamlit_app.py).

Run locally with: streamlit run streamlit_app.py
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

st.set_page_config(page_title="Hydro-TGT: Fraud Detection", page_icon="🔎")

CHECKPOINT_PATH = "checkpoints/hydro_tgt.pt"
GRAPH_PATH = "dataset/graph.pt"


# ----------------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = None
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _pos(self, seq_len, device):
        if self.pos_embedding is None or self.pos_embedding.shape[1] != seq_len + 1:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, self.d_model, device=device))
        return self.pos_embedding

    def forward(self, temporal_x):
        b, seq_len, _ = temporal_x.shape
        h = self.input_proj(temporal_x)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1) + self._pos(seq_len, h.device)
        return self.encoder(h)[:, 0, :]


class HydroTGT(nn.Module):
    def __init__(self, static_in_dim, gnn_hidden=64, gnn_out=64, temporal_d_model=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.gnn = GNNEncoder(static_in_dim, gnn_hidden, gnn_out, dropout=dropout)
        self.temporal = TemporalTransformer(input_dim=2, d_model=temporal_d_model)
        fused_dim = gnn_out + temporal_d_model
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, temporal_x):
        fused = torch.cat([self.gnn(x, edge_index), self.temporal(temporal_x)], dim=1)
        return self.classifier(fused)


# ----------------------------------------------------------------------------
# Synthetic data + graph building (used only if no real checkpoint is committed)
# ----------------------------------------------------------------------------
def generate_synthetic(n_wallets=300, n_tx=3000, fraud_ratio=0.08, seed=42):
    rng = np.random.default_rng(seed)
    wallet_ids = [f"w{i}" for i in range(n_wallets)]
    nodes = pd.DataFrame({"wallet_id": wallet_ids, "first_seen_ts": rng.integers(0, 1_000_000, size=n_wallets)})
    fraud_wallets = set(rng.choice(wallet_ids, size=int(n_wallets * fraud_ratio), replace=False))
    labels = pd.DataFrame({"wallet_id": wallet_ids, "label": [1 if w in fraud_wallets else 0 for w in wallet_ids]})
    transactions = pd.DataFrame({
        "tx_id": [f"tx{i}" for i in range(n_tx)],
        "src_wallet": rng.choice(wallet_ids, size=n_tx),
        "dst_wallet": rng.choice(wallet_ids, size=n_tx),
        "amount": rng.exponential(scale=50.0, size=n_tx),
        "timestamp": np.sort(rng.integers(0, 1_000_000, size=n_tx)),
    })
    return nodes, transactions, labels


def build_static_features(nodes, transactions):
    wallets = nodes["wallet_id"].tolist()
    idx = {w: i for i, w in enumerate(wallets)}
    n = len(wallets)
    in_deg, out_deg, sent, recv, cnt = (np.zeros(n) for _ in range(5))
    for _, row in transactions.iterrows():
        s, d, amt = idx.get(row["src_wallet"]), idx.get(row["dst_wallet"]), row["amount"]
        if s is not None:
            out_deg[s] += 1; sent[s] += amt; cnt[s] += 1
        if d is not None:
            in_deg[d] += 1; recv[d] += amt; cnt[d] += 1
    avg_amt = np.divide(sent + recv, np.maximum(cnt, 1), out=np.zeros(n), where=cnt > 0)
    return np.stack([in_deg, out_deg, sent, recv, avg_amt, cnt], axis=1)


def build_temporal_sequences(nodes, transactions, seq_len=16):
    wallets = nodes["wallet_id"].tolist()
    idx = {w: i for i, w in enumerate(wallets)}
    events = {w: [] for w in wallets}
    for _, row in transactions.sort_values("timestamp").iterrows():
        for w in (row["src_wallet"], row["dst_wallet"]):
            if w in events:
                events[w].append((row["amount"], row["timestamp"]))
    seqs = np.zeros((len(wallets), seq_len, 2), dtype=np.float32)
    for w, evs in events.items():
        if not evs:
            continue
        evs = evs[-seq_len:]
        amounts = [e[0] for e in evs]
        times = [e[1] for e in evs]
        deltas = [0.0] + [times[i] - times[i - 1] for i in range(1, len(times))]
        arr = np.stack([amounts, deltas], axis=1)
        seqs[idx[w], -len(arr):, :] = arr
    return seqs


def build_graph(nodes, transactions, labels, seq_len=16):
    wallets = nodes["wallet_id"].tolist()
    idx = {w: i for i, w in enumerate(wallets)}
    n = len(wallets)

    static = build_static_features(nodes, transactions)
    static = (static - static.mean(axis=0)) / (static.std(axis=0) + 1e-6)
    x = torch.tensor(static, dtype=torch.float)
    temporal_x = torch.tensor(build_temporal_sequences(nodes, transactions, seq_len), dtype=torch.float)

    src = transactions["src_wallet"].map(idx).values
    dst = transactions["dst_wallet"].map(idx).values
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    label_map = dict(zip(labels["wallet_id"], labels["label"]))
    y = torch.tensor([label_map.get(w, -1) for w in wallets], dtype=torch.long)

    labeled = (y != -1).nonzero(as_tuple=True)[0]
    perm = labeled[torch.randperm(len(labeled))]
    n_train, n_val = int(0.7 * len(perm)), int(0.15 * len(perm))
    train_mask, val_mask, test_mask = (torch.zeros(n, dtype=torch.bool) for _ in range(3))
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.temporal_x = temporal_x
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    data.wallet_ids = wallets
    return data


# ----------------------------------------------------------------------------
# Bootstrap: load a committed model if present, otherwise train a quick demo one
# ----------------------------------------------------------------------------
@st.cache_resource
def load_or_train():
    if os.path.exists(GRAPH_PATH) and os.path.exists(CHECKPOINT_PATH):
        data = torch.load(GRAPH_PATH, weights_only=False)
        model = HydroTGT(static_in_dim=data.x.shape[1])
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model.eval()
        return data, model, False

    nodes, transactions, labels = generate_synthetic()
    data = build_graph(nodes, transactions, labels)
    model = HydroTGT(static_in_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    train_y = data.y[data.train_mask]
    n_pos, n_neg = (train_y == 1).sum().item(), (train_y == 0).sum().item()
    weights = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 1.0)])

    model.train()
    for _ in range(25):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.temporal_x)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=weights)
        loss.backward()
        optimizer.step()
    model.eval()
    return data, model, True


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🔎 Hydro-TGT — Blockchain Fraud Detection")
st.caption("GNN + Transformer hybrid model for flow-aware fraud detection")

with st.spinner("Loading model (first run trains a quick demo model — ~30s)..."):
    data, model, is_demo = load_or_train()
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index, data.temporal_x), dim=1)[:, 1]

if is_demo:
    st.info(
        "Running on an auto-generated synthetic wallet graph (no trained checkpoint was "
        "found in the repo). See README for training on real data and committing the checkpoint.",
        icon="ℹ️",
    )

wallet_index = {w: i for i, w in enumerate(data.wallet_ids)}

tab1, tab2 = st.tabs(["Lookup a wallet", "Top risky wallets"])

with tab1:
    wallet_id = st.text_input("Wallet ID", placeholder="e.g. w42")
    if st.button("Check", type="primary") and wallet_id:
        if wallet_id in wallet_index:
            prob = float(probs[wallet_index[wallet_id]])
            st.metric("Fraud probability", f"{prob:.1%}")
            st.error("⚠️ Flagged as likely fraudulent") if prob > 0.5 else st.success("✅ Looks normal")
        else:
            st.warning(f"Unknown wallet_id: {wallet_id}")

with tab2:
    k = st.slider("Show top N riskiest wallets", 5, 50, 10)
    top_idx = torch.topk(probs, k).indices.tolist()
    st.table({
        "Wallet ID": [data.wallet_ids[i] for i in top_idx],
        "Fraud probability": [f"{float(probs[i]):.1%}" for i in top_idx],
    })

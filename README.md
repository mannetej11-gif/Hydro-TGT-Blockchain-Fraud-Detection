# Hydro-TGT: Flow-Aware Blockchain Fraud Detection using GNN-Transformer

A hybrid deep learning model that detects fraudulent blockchain transactions by modeling
transaction history as a graph and learning:

- **Spatial relationships** between wallets using a **Graph Neural Network** (GraphSAGE)
- **Temporal transaction patterns** per wallet using a **Transformer encoder**

The two embeddings are fused and passed through a classifier that outputs `fraud` / `normal`.

## Architecture

```
Transaction Dataset
        │
        ▼
Data Preprocessing  ──►  clean_data.py
        │
        ▼
Graph Construction  ──►  graph_builder.py
(Nodes = Wallets, Edges = Transactions)
        │
        ├──────────────┬───────────────┐
        ▼              ▼               │
  GraphSAGE/GAT   Temporal Transformer  │
  (models/gnn.py) (models/transformer.py)
        │              │               │
        └──────┬───────┘               │
               ▼                       │
     Fusion + Classifier ◄─────────────┘
     (models/classifier.py)
               │
               ▼
        Fraud / Normal
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| GNN | PyTorch Geometric (GraphSAGE) |
| Temporal modeling | PyTorch `nn.TransformerEncoder` |
| Dataset | Elliptic Bitcoin Dataset (or any wallet-transaction CSV with the same schema) |
| Data processing | Pandas, NumPy |
| Evaluation | Scikit-learn |
| Demo | Streamlit |

## Folder Structure

```
Hydro-TGT/
├── dataset/                 # raw + processed data (not committed — see dataset/README.md)
├── models/
│   ├── gnn.py                # GraphSAGE spatial encoder
│   ├── transformer.py        # Temporal transformer encoder
│   └── classifier.py         # Fusion + classification head (full Hydro-TGT model)
├── preprocessing/
│   ├── clean_data.py         # Cleaning, dedup, label mapping
│   ├── graph_builder.py      # Builds a PyG graph from cleaned transactions
│   └── feature_engineering.py# Node stats + per-node temporal transaction sequences
├── training/
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation on held-out set
│   └── metrics.py             # Accuracy, Precision, Recall, F1, ROC-AUC
├── notebooks/
│   └── Hydro-TGT.ipynb        # End-to-end walkthrough (Colab-friendly)
├── app/
│   ├── app.py                 # Streamlit demo
│   └── predict.py             # Inference helper used by the app
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/mannetej11-gif/Hydro-TGT-Blockchain-Fraud-Detection.git
cd Hydro-TGT-Blockchain-Fraud-Detection
pip install -r requirements.txt
```

> ⚠️ `torch` and `torch-geometric` aren't preinstalled almost anywhere by default — since you're working from a phone, run this project in **Google Colab** (free GPU, works entirely in-browser, no local installs needed). Open `notebooks/Hydro-TGT.ipynb` in Colab and run the setup cell first.

## Dataset

This project expects three CSV files in `dataset/`:

- `nodes.csv` — `wallet_id, first_seen_ts, ...`
- `transactions.csv` — `tx_id, src_wallet, dst_wallet, amount, timestamp`
- `labels.csv` — `wallet_id, label` (`1` = fraud, `0` = normal, `-1` = unknown)

The **Elliptic Bitcoin Dataset** (Kaggle) matches this shape closely and is the recommended
starting dataset — see `dataset/README.md` for the exact download + remapping steps.

## Running the Pipeline

```bash
# 1. Clean raw data
python preprocessing/clean_data.py --input dataset/raw --output dataset/clean

# 2. Build the graph + temporal features
python preprocessing/graph_builder.py --input dataset/clean --output dataset/graph.pt

# 3. Train
python training/train.py --graph dataset/graph.pt --epochs 50 --out checkpoints/hydro_tgt.pt

# 4. Evaluate
python training/evaluate.py --graph dataset/graph.pt --checkpoint checkpoints/hydro_tgt.pt

# 5. Demo
streamlit run app/app.py
```

## Development Roadmap

- [x] Repo scaffold, architecture, tech stack
- [ ] **Phase 1** — Environment setup, dataset download & inspection
- [ ] **Phase 2** — Cleaning, graph construction, feature engineering
- [ ] **Phase 3** — Implement GNN, Transformer, fuse into hybrid model
- [ ] **Phase 4** — Train, evaluate, visualize results
- [ ] **Phase 5** — Streamlit demo + optional deployment

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC — see `training/metrics.py`.

## Resume Line

> Designed a hybrid GNN-Transformer model for blockchain fraud detection, modeling wallet
> transactions as a graph to capture structural relationships and using temporal attention
> to identify evolving fraud patterns; evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC.

## License

MIT — see `LICENSE`.

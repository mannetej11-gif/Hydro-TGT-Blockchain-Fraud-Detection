"""
app.py
------
Streamlit demo for Hydro-TGT.

Run with:
    streamlit run app/app.py
"""
import streamlit as st
from predict import HydroTGTPredictor
from bootstrap import ensure_demo_artifacts

st.set_page_config(page_title="Hydro-TGT: Fraud Detection", page_icon="🔎")
st.title("🔎 Hydro-TGT — Blockchain Fraud Detection")
st.caption("GNN + Transformer hybrid model for flow-aware fraud detection")

WALLET_IDS_PATH = "dataset/clean/nodes.csv"


@st.cache_resource
def load_predictor():
    # If no committed checkpoint/graph exist (e.g. first deploy), this trains a
    # quick demo model on synthetic data instead of erroring out.
    graph_path, checkpoint_path = ensure_demo_artifacts()
    return HydroTGTPredictor(graph_path, checkpoint_path, WALLET_IDS_PATH)


with st.spinner("Loading model (first run trains a quick demo model — ~30s)..."):
    predictor = load_predictor()

st.info(
    "This demo is running on an auto-generated synthetic wallet graph unless a real "
    "trained checkpoint has been committed to the repo. See README for training on real data.",
    icon="ℹ️",
)

tab1, tab2 = st.tabs(["Lookup a wallet", "Top risky wallets"])

with tab1:
    wallet_id = st.text_input("Wallet ID", placeholder="e.g. w42")
    if st.button("Check", type="primary") and wallet_id:
        try:
            prob = predictor.predict_by_wallet_id(wallet_id)
            st.metric("Fraud probability", f"{prob:.1%}")
            if prob > 0.5:
                st.error("⚠️ Flagged as likely fraudulent")
            else:
                st.success("✅ Looks normal")
        except ValueError as e:
            st.warning(str(e))

with tab2:
    k = st.slider("Show top N riskiest wallets", 5, 50, 10)
    results = predictor.top_risky_wallets(k)
    st.table(
        {
            "Node index": [r[0] for r in results],
            "Fraud probability": [f"{r[1]:.1%}" for r in results],
        }
    )

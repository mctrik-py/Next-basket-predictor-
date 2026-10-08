import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Next Basket Predictor", page_icon="🛒", layout="wide", initial_sidebar_state="collapsed")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "domain" not in st.session_state:
    st.session_state.domain = None
if "input_mode" not in st.session_state:
    st.session_state.input_mode = None
if "extracted_items" not in st.session_state:
    st.session_state.extracted_items = None
if "confirmed_items" not in st.session_state:
    st.session_state.confirmed_items = None
if "preds" not in st.session_state:
    st.session_state.preds = None
if "manual_selection" not in st.session_state:
    st.session_state.manual_selection = []

THEMES = {
    "Light": {"bg": "#f8f9fc", "card": "#ffffff", "text": "#1f2937", "muted": "#6b7280", "border": "#e5e7eb", "accent": "#6366f1", "accent2": "#8b5cf6", "success": "#10b981", "warning": "#f59e0b", "hero_from": "#667eea", "hero_to": "#764ba2"},
    "Dark": {"bg": "#0f172a", "card": "#1e293b", "text": "#f1f5f9", "muted": "#94a3b8", "border": "#334155", "accent": "#818cf8", "accent2": "#a78bfa", "success": "#34d399", "warning": "#fbbf24", "hero_from": "#4c1d95", "hero_to": "#7c3aed"},
}

def apply_theme_css(theme_name):
    t = THEMES.get(theme_name, THEMES["Light"])
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {t["bg"]}; }}
        .main .block-container {{ padding-top: 1rem; max-width: 1200px; }}
        .hero {{ background: linear-gradient(135deg, {t["hero_from"]} 0%, {t["hero_to"]} 100%); padding: 2rem; border-radius: 16px; text-align: center; color: white; margin-bottom: 2rem; }}
        .hero h1 {{ color: white; font-size: 2.4rem; margin: 0; font-weight: 800; }}
        .hero p {{ color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1rem; }}
        .step-header {{ font-size: 1.1rem; font-weight: 700; color: {t["text"]}; margin: 1.5rem 0 0.75rem 0; display: flex; align-items: center; gap: 0.5rem; }}
        .step-num {{ display: inline-flex; width: 32px; height: 32px; background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]}); color: white; border-radius: 50%; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; }}
        .pred-card {{ background: {t["card"]}; padding: 1rem 1.25rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 0.6rem; border-left: 4px solid {t["accent"]}; }}
        .pred-card.top {{ border-left: 4px solid {t["warning"]}; }}
        .pred-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .pred-rank {{ font-size: 1.2rem; font-weight: 800; color: {t["accent"]}; min-width: 40px; }}
        .pred-card.top .pred-rank {{ color: {t["warning"]}; }}
        .pred-name {{ flex: 1; font-size: 1rem; font-weight: 600; color: {t["text"]}; padding: 0 1rem; }}
        .pred-prob {{ font-size: 1.1rem; font-weight: 800; color: {t["success"]}; min-width: 70px; text-align: right; }}
        .pred-card.top .pred-prob {{ color: {t["warning"]}; }}
        .progress-track {{ height: 6px; background: {t["border"]}; border-radius: 3px; margin-top: 6px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, {t["accent"]}, {t["accent2"]}); border-radius: 3px; }}
        .info-box {{ background: {t["card"]}; border-left: 4px solid {t["accent"]}; padding: 1rem 1.25rem; border-radius: 8px; color: {t["text"]}; margin: 1rem 0; }}
        .warn-box {{ background: {t["card"]}; border-left: 4px solid {t["warning"]}; padding: 1rem 1.25rem; border-radius: 8px; color: {t["text"]}; margin: 1rem 0; }}
        .succ-box {{ background: {t["card"]}; border-left: 4px solid {t["success"]}; padding: 1rem 1.25rem; border-radius: 8px; color: {t["text"]}; margin: 1rem 0; }}
        .stat-card {{ background: {t["card"]}; padding: 1.25rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; border-top: 3px solid {t["accent"]}; }}
        .stat-value {{ font-size: 1.8rem; font-weight: 800; color: {t["accent"]}; margin: 0; }}
        .stat-label {{ font-size: 0.8rem; color: {t["muted"]}; margin-top: 0.25rem; text-transform: uppercase; }}
        .item-chip {{ display: inline-block; padding: 0.4rem 0.9rem; background: {t["accent"]}; color: white; border-radius: 20px; margin: 0.25rem; font-size: 0.9rem; font-weight: 500; animation: popIn 0.3s ease-out; }}
        @keyframes popIn {{ 0% {{ transform: scale(0.6); opacity: 0; }} 100% {{ transform: scale(1); opacity: 1; }} }}
        @keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        @keyframes slideUp {{ 0% {{ transform: translateY(20px); opacity: 0; }} 100% {{ transform: translateY(0); opacity: 1; }} }}
        .cart-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, {t["success"]}, #059669);
            color: white; border-radius: 50%;
            width: 36px; height: 36px;
            font-weight: 800; font-size: 1rem;
            margin-left: 10px;
            animation: pulse 1.5s ease-in-out infinite;
            box-shadow: 0 4px 12px rgba(16,185,129,0.4);
        }}
        .pred-card {{ animation: slideUp 0.5s ease-out; }}
        .pred-ring {{
            width: 60px; height: 60px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            background: conic-gradient({t["accent"]} var(--p), {t["border"]} 0);
            position: relative;
            animation: popIn 0.6s ease-out;
        }}
        .pred-ring::before {{
            content: '';
            width: 48px; height: 48px;
            border-radius: 50%;
            background: {t["card"]};
            position: absolute;
        }}
        .pred-ring-text {{
            position: relative;
            font-weight: 800;
            color: {t["accent"]};
            font-size: 0.75rem;
        }}
        .category-icon {{
            font-size: 1.5rem;
            margin-right: 0.5rem;
            display: inline-block;
        }}
        .spinner {{
            display: inline-block;
            width: 20px; height: 20px;
            border: 3px solid {t["border"]};
            border-top-color: {t["accent"]};
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
        }}
        .confetti {{
            font-size: 2rem;
            animation: popIn 0.5s ease-out;
        }}
        div[data-testid="stRadio"] label {{ color: {t["text"]} !important; }}
        .stButton > button {{ background: linear-gradient(135deg, {t["accent"]}, {t["accent2"]}); color: white !important; border: none; border-radius: 10px; font-weight: 700; padding: 0.7rem 1.5rem; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme_css(st.session_state.theme)

@st.cache_resource
def load_all():
    DEVICE = torch.device("cpu")
    with open("data/vocab/product_to_idx.json", "r") as f:
        product_to_idx = json.load(f)
    with open("data/vocab/idx_to_product.json", "r") as f:
        idx_to_product = {int(k): v for k, v in json.load(f).items()}
    with open("data/vocab/bnf_to_idx.json", "r") as f:
        bnf_to_idx = json.load(f)

    products_df = pd.read_csv("data/instacart/products.csv")
    product_id_to_name = dict(zip(products_df["product_id"], products_df["product_name"]))
    aisles_df = pd.read_csv("data/instacart/aisles.csv")
    products_with_aisles = products_df.merge(aisles_df, on="aisle_id")
    aisle_to_products = products_with_aisles.groupby("aisle")["product_name"].apply(list).to_dict()

    bnf_df = pd.read_csv("data/processed/nhs_lookup.csv")
    bnf_to_name_full = dict(zip(bnf_df["BNFItemCode"].astype(str), bnf_df["BNFItemDescription"]))

    BNF_CHAPTERS = {
        "01": "Gastro-intestinal", "02": "Cardiovascular", "03": "Respiratory",
        "04": "Central Nervous System", "05": "Infections", "06": "Endocrine",
        "07": "Genito-urinary", "08": "Malignant Disease", "09": "Nutrition & Blood",
        "10": "Musculoskeletal", "11": "Eye", "12": "Ear/Nose/Throat", "13": "Skin"
    }
    bnf_by_chapter = {}
    for code, name in bnf_to_name_full.items():
        chapter = str(code)[:2]
        if chapter in BNF_CHAPTERS:
            bnf_by_chapter.setdefault(BNF_CHAPTERS[chapter], []).append(name)

    class LFIGLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.slope = nn.Parameter(torch.zeros(d_model))
            self.intercept = nn.Parameter(torch.zeros(d_model))
            self.spread = nn.Parameter(torch.ones(d_model))
        def forward(self, x):
            B, T, D = x.shape
            t = torch.arange(T, dtype=torch.float32, device=x.device).view(1, T, 1)
            trend = self.slope.view(1, 1, D) * t + self.intercept.view(1, 1, D)
            trend = trend.expand(B, T, D)
            fuzzy = torch.exp(-((x - trend) ** 2) / (2 * self.spread.view(1, 1, D) ** 2 + 1e-6))
            return trend, fuzzy

    class TLFIGTransformer(nn.Module):
        def __init__(self, num_items, d_model=64, n_heads=4, n_layers=2):
            super().__init__()
            self.input_proj = nn.Linear(num_items, d_model)
            self.lfig = LFIGLayer(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
            self.trend_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.feature_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, num_items))
        def forward(self, x):
            h = self.input_proj(x)
            trend, fuzzy = self.lfig(h)
            t_out = self.trend_encoder(trend)
            f_out = self.feature_encoder(h * fuzzy)
            combined = torch.cat([t_out[:, -1, :], f_out[:, -1, :]], dim=-1)
            return torch.sigmoid(self.head(combined))

    def detect_arch(state_dict):
        d_model = state_dict["input_proj.weight"].shape[0]
        n_layers = sum(1 for k in state_dict.keys() if k.startswith("trend_encoder.layers.") and k.endswith("norm1.weight"))
        if n_layers == 0:
            n_layers = 1
        n_heads = 4 if d_model == 64 else 2
        return d_model, n_heads, n_layers

    gm_state = torch.load("models/grocery_tlfig.pt", map_location=DEVICE)
    gm_d, gm_h, gm_l = detect_arch(gm_state)
    gm = TLFIGTransformer(num_items=len(product_to_idx), d_model=gm_d, n_heads=gm_h, n_layers=gm_l).to(DEVICE)
    gm.load_state_dict(gm_state)
    gm.eval()

    pm_state = torch.load("models/pharmacy_tlfig.pt", map_location=DEVICE)
    pm_d, pm_h, pm_l = detect_arch(pm_state)
    pm = TLFIGTransformer(num_items=len(bnf_to_idx), d_model=pm_d, n_heads=pm_h, n_layers=pm_l).to(DEVICE)
    pm.load_state_dict(pm_state)
    pm.eval()

    return gm, pm, product_to_idx, idx_to_product, bnf_to_idx, product_id_to_name, bnf_to_name_full, aisle_to_products, bnf_by_chapter

grocery_model, pharmacy_model, product_to_idx, idx_to_product, bnf_to_idx, product_id_to_name, bnf_to_name_full, aisle_to_products, bnf_by_chapter = load_all()

import sys
sys.path.append("src")
from llm_extractor import extract_from_text

def find_product_matches(item_names, max_per_item=2):
    matched_ids = []
    for name in item_names:
        name_lower = name.lower().strip()
        tokens = [t for t in name_lower.split() if len(t) > 2]
        if not tokens:
            continue
        candidates = []
        for pid in product_to_idx.keys():
            pname = product_id_to_name.get(int(pid), "").lower()
            if name_lower == pname:
                score = 1000
            elif pname.startswith(name_lower):
                score = 500
            elif name_lower in pname:
                score = 100 - len(pname)
            elif tokens and all(t in pname for t in tokens):
                score = 50 + sum(1 for t in tokens if t in pname) * 5
            else:
                continue
            score -= len(pname) // 10
            candidates.append((score, int(pid)))
        candidates.sort(reverse=True)
        matched_ids.extend(pid for _, pid in candidates[:max_per_item])
    return list(dict.fromkeys(matched_ids))

def predict_grocery_named(item_names, top_k=10):
    vec = np.zeros(len(product_to_idx), dtype=np.float32)
    matched_pids = find_product_matches(item_names)
    for pid in matched_pids:
        key = str(pid)
        if key in product_to_idx:
            vec[product_to_idx[key]] = 1.0
    if vec.sum() == 0:
        vec[:] = 0.5
    seq = np.tile(vec, (3, 1))
    seq_t = torch.tensor(seq).unsqueeze(0)
    with torch.no_grad():
        pred = grocery_model(seq_t).squeeze(0).numpy()
    top_idx = np.argsort(pred)[-(top_k * 4):][::-1]
    results = []
    seen_stems = set()
    for i in top_idx:
        name = product_id_to_name.get(int(idx_to_product.get(int(i))), f"Product {i}")
        stem = name.lower().rstrip("s").strip()
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        results.append({"item": name, "probability": float(pred[i])})
        if len(results) >= top_k:
            break
    return results

def predict_pharmacy_named(item_names, top_k=10):
    vec = np.zeros(len(bnf_to_idx), dtype=np.float32)
    for name in item_names:
        name_lower = name.lower().strip()
        tokens = [t for t in name_lower.split() if len(t) > 2]
        for code, idx in bnf_to_idx.items():
            bnf_name = bnf_to_name_full.get(str(code), "").lower()
            if name_lower in bnf_name or bnf_name in name_lower or (tokens and all(t in bnf_name for t in tokens)):
                vec[idx] = 1.0
                break
    if vec.sum() == 0:
        vec[:] = 0.5
    seq = np.tile(vec, (3, 1))
    seq_t = torch.tensor(seq).unsqueeze(0)
    with torch.no_grad():
        pred = pharmacy_model(seq_t).squeeze(0).numpy()
    top_idx = np.argsort(pred)[-top_k:][::-1]
    results = []
    for i in top_idx:
        code = next((k for k, v in bnf_to_idx.items() if v == int(i)), None)
        name = bnf_to_name_full.get(str(code), f"BNF {code}") if code else f"Idx {i}"
        results.append({"item": name[:80], "probability": float(pred[i])})
    return results

def generate_pdf(domain, items, predictions):
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/report_{domain}_{int(time.time())}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Next Basket Prediction Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Domain: {domain.upper()}")
    c.drawString(50, height - 100, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 140, "Input Items:")
    y = height - 160
    c.setFont("Helvetica", 10)
    for item in items[:10]:
        c.drawString(70, y, f"- {item}")
        y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 20, "Top Predictions:")
    y -= 40
    c.setFont("Helvetica", 10)
    for i, p in enumerate(predictions[:10], 1):
        c.drawString(70, y, f"{i}. {p['item']}  ({p['probability']*100:.1f}%)")
        y -= 15
    c.save()
    return filename

top_left, top_right = st.columns([6, 1])
with top_right:
    theme_choice = st.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, label_visibility="collapsed")
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

st.markdown("""<div class="hero"><h1>🛒 Next Basket Predictor</h1><p>GPBoost + TLFIG-Transformer</p></div>""", unsafe_allow_html=True)

st.markdown('<div class="step-header"><span class="step-num">1</span>Select Domain</div>', unsafe_allow_html=True)
domain_col1, domain_col2 = st.columns(2)
with domain_col1:
    if st.button("🛒 Grocery", use_container_width=True, type="primary" if st.session_state.domain == "Grocery" else "secondary"):
        st.session_state.domain = "Grocery"
        st.session_state.preds = None
        st.session_state.extracted_items = None
        st.session_state.confirmed_items = None
        st.session_state.manual_selection = []
        st.session_state.input_mode = None
        st.rerun()
with domain_col2:
    if st.button("💊 Pharmacy", use_container_width=True, type="primary" if st.session_state.domain == "Pharmacy" else "secondary"):
        st.session_state.domain = "Pharmacy"
        st.session_state.preds = None
        st.session_state.extracted_items = None
        st.session_state.confirmed_items = None
        st.session_state.manual_selection = []
        st.session_state.input_mode = None
        st.rerun()

if st.session_state.domain:
    st.markdown('<div class="step-header"><span class="step-num">2</span>Choose Input Method</div>', unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("📝 Describe in Plain Text", use_container_width=True, type="primary" if st.session_state.input_mode == "text" else "secondary"):
            st.session_state.input_mode = "text"
            st.session_state.extracted_items = None
            st.session_state.confirmed_items = None
            st.rerun()
    with mode_col2:
        if st.button("📋 Select from Categories", use_container_width=True, type="primary" if st.session_state.input_mode == "manual" else "secondary"):
            st.session_state.input_mode = "manual"
            st.session_state.extracted_items = None
            st.session_state.confirmed_items = None
            st.session_state.manual_selection = []
            st.rerun()

if st.session_state.domain and st.session_state.input_mode:
    st.markdown('<div class="step-header"><span class="step-num">3</span>Enter Your Items</div>', unsafe_allow_html=True)

    if st.session_state.input_mode == "text":
        user_text = st.text_area("Describe what you need:", placeholder="e.g., I need milk, eggs, bread, and bananas", height=100, key="input_text")
        if st.button("🔍 Extract Items", type="primary", use_container_width=True):
            if not user_text.strip():
                st.markdown('<div class="warn-box">⚠️ Please enter some text</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Extracting items with LLM..."):
                    domain_key = "grocery" if st.session_state.domain == "Grocery" else "pharmacy"
                    result = extract_from_text(user_text, domain_key)
                if result["status"] == "success":
                    items = [i.get("normalized") or i.get("raw") for i in result["data"].get("items", [])]
                    st.session_state.extracted_items = items
                    st.rerun()
                else:
                    st.markdown(f'<div class="warn-box">⚠️ LLM failed: {result["message"]}. Please use manual input.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">📋 <b>Pick items from any category. Your cart persists across category changes.</b></div>', unsafe_allow_html=True)

        cart_count = len(st.session_state.manual_selection)
        st.markdown(f'<h3 style="color: var(--text); display: inline;">🛒 Your Cart</h3><span class="cart-badge">{cart_count}</span>', unsafe_allow_html=True)

        if st.session_state.domain == "Grocery":
            st.markdown("**⚡ Quick Picks — Common Grocery Bundles:**")
            qp_cols = st.columns(4)
            quick_picks = {
                "🥛 Breakfast": ["milk", "bread", "eggs", "butter"],
                "🍝 Italian": ["pasta", "tomato sauce", "cheese", "garlic"],
                "🥗 Salad": ["lettuce", "tomato", "cucumber", "olive oil"],
                "🍎 Snacks": ["banana", "apple", "yogurt", "nuts"]
            }
        else:
            st.markdown("**⚡ Quick Picks — Common Therapeutic Bundles:**")
            qp_cols = st.columns(4)
            quick_picks = {
                "❤️ Cardiovascular": ["atorvastatin", "amlodipine", "lisinopril", "aspirin"],
                "🩸 Diabetes": ["metformin", "insulin", "gliclazide", "test strips"],
                "🫁 Respiratory": ["salbutamol", "beclometasone", "prednisolone", "inhaler"],
                "🦴 Pain & Bone": ["paracetamol", "ibuprofen", "calcium", "vitamin d"]
            }

        for idx, (label, items) in enumerate(quick_picks.items()):
            with qp_cols[idx]:
                if st.button(label, use_container_width=True, key=f"qp_{st.session_state.domain}_{idx}"):
                    for item in items:
                        if item not in st.session_state.manual_selection:
                            st.session_state.manual_selection.append(item)
                    st.toast(f"Added {label} bundle", icon="⚡")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.domain == "Grocery":
            aisle_options = sorted(aisle_to_products.keys())
            selected_aisle = st.selectbox("Step A — Choose category:", ["-- Select a category --"] + aisle_options, key="grocery_aisle")
            if selected_aisle != "-- Select a category --":
                product_options = sorted(aisle_to_products[selected_aisle])
                selected_item = st.selectbox(f"Step B — Pick item from {selected_aisle}:", ["-- Pick an item --"] + product_options, key=f"pick_{selected_aisle}")
                if selected_item != "-- Pick an item --":
                    col_add, col_clear = st.columns([2, 1])
                    with col_add:
                        if st.button(f"➕ Add '{selected_item}' to Cart", type="primary", use_container_width=True):
                            if selected_item not in st.session_state.manual_selection:
                                st.session_state.manual_selection.append(selected_item)
                                st.rerun()
                            else:
                                st.markdown('<div class="warn-box">⚠️ Already in cart</div>', unsafe_allow_html=True)
                    with col_clear:
                        if st.button("🗑️ Clear All", use_container_width=True):
                            st.session_state.manual_selection = []
                            st.rerun()
        else:
            chapter_options = sorted(bnf_by_chapter.keys())
            selected_chapter = st.selectbox("Step A — Choose drug category:", ["-- Select a category --"] + chapter_options, key="pharmacy_chapter")
            if selected_chapter != "-- Select a category --":
                drug_options = sorted(set(bnf_by_chapter[selected_chapter]))[:500]
                selected_drug = st.selectbox(f"Step B — Pick drug from {selected_chapter}:", ["-- Pick a drug --"] + drug_options, key=f"pick_drug_{selected_chapter}")
                if selected_drug != "-- Pick a drug --":
                    col_add, col_clear = st.columns([2, 1])
                    with col_add:
                        if st.button(f"➕ Add to Cart", type="primary", use_container_width=True):
                            if selected_drug not in st.session_state.manual_selection:
                                st.session_state.manual_selection.append(selected_drug)
                                st.rerun()
                            else:
                                st.markdown('<div class="warn-box">⚠️ Already in cart</div>', unsafe_allow_html=True)
                    with col_clear:
                        if st.button("🗑️ Clear All", use_container_width=True):
                            st.session_state.manual_selection = []
                            st.rerun()

        if st.session_state.manual_selection:
            st.markdown('<div class="info-box">🛒 <b>Your cart:</b></div>', unsafe_allow_html=True)
            for idx, item in enumerate(st.session_state.manual_selection):
                col_item, col_remove = st.columns([5, 1])
                with col_item:
                    st.markdown(f'<span class="item-chip">✓ {item[:70]}</span>', unsafe_allow_html=True)
                with col_remove:
                    if st.button("❌", key=f"remove_{idx}", use_container_width=True):
                        st.session_state.manual_selection.pop(idx)
                        st.toast(f"Removed: {item}", icon="🗑️")
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Confirm Items", type="primary", use_container_width=True):
                    st.session_state.extracted_items = st.session_state.manual_selection.copy()
                    st.toast(f"Confirmed {len(st.session_state.manual_selection)} items!", icon="✅")
                    st.rerun()
            with col_b:
                if st.button("🔄 Reset Cart", use_container_width=True):
                    st.session_state.manual_selection = []
                    st.toast("Cart cleared", icon="🔄")
                    st.rerun()

if st.session_state.extracted_items:
    st.markdown('<div class="step-header"><span class="step-num">4</span>Confirm Your Items</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📋 <b>Items detected:</b> Review below and edit if needed before predicting.</div>', unsafe_allow_html=True)
    chips_html = "".join([f'<span class="item-chip">✓ {item}</span>' for item in st.session_state.extracted_items])
    st.markdown(f'<div style="margin: 1rem 0;">{chips_html}</div>', unsafe_allow_html=True)
    edited_text = st.text_area("Edit items (comma-separated):", value=", ".join(st.session_state.extracted_items), height=80, key="edit_items")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Confirm & Predict", type="primary", use_container_width=True):
            final_items = [x.strip() for x in edited_text.split(",") if x.strip()]
            if not final_items:
                st.markdown('<div class="warn-box">⚠️ Please enter at least one item</div>', unsafe_allow_html=True)
            else:
                progress_text = "🎯 Running TLFIG-Transformer..."
                my_bar = st.progress(0, text=progress_text)
                import time as _time
                for percent_complete in range(100):
                    _time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                preds = predict_grocery_named(final_items) if st.session_state.domain == "Grocery" else predict_pharmacy_named(final_items)
                my_bar.empty()
                st.session_state.confirmed_items = final_items
                st.session_state.preds = preds
                st.toast(f"Predicted {len(preds)} items!", icon="🎯")
                st.rerun()
    with col_b:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.extracted_items = None
            st.session_state.confirmed_items = None
            st.session_state.preds = None
            st.session_state.manual_selection = []
            st.rerun()

if st.session_state.preds:
    st.markdown('<div class="step-header"><span class="step-num">5</span>Top Next Items You Might Purchase</div>', unsafe_allow_html=True)
    st.markdown('<div class="confetti">🎉</div>', unsafe_allow_html=True)
    st.markdown('<div class="succ-box">✨ <b>Prediction complete!</b> Here are your top items:</div>', unsafe_allow_html=True)

    for i, p in enumerate(st.session_state.preds, 1):
        prob_pct = p["probability"] * 100
        card_cls = "pred-card top" if i == 1 else "pred-card"
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        st.markdown(f"""
        <div class="{card_cls}">
            <div class="pred-row">
                <div class="pred-rank">{medal}</div>
                <div class="pred-name">{p["item"]}</div>
                <div class="pred-ring" style="--p: {prob_pct}%">
                    <span class="pred-ring-text">{prob_pct:.0f}%</span>
                </div>
            </div>
            <div class="progress-track"><div class="progress-fill" style="width:{prob_pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # Expandable explanation
    with st.expander("💡 Why these predictions? Click to see details"):
        top_item = st.session_state.preds[0]["item"]
        top_prob = st.session_state.preds[0]["probability"] * 100
        avg_prob = np.mean([p["probability"] for p in st.session_state.preds]) * 100
        confirmed = st.session_state.confirmed_items or st.session_state.selected_items or []
        inputs_str = ", ".join(confirmed[:5]) if confirmed else "N/A"
        st.markdown(f"""
        - **Top prediction:** {top_item} with **{top_prob:.1f}%** confidence
        - **Average confidence across top-10:** {avg_prob:.1f}%
        - **Based on inputs:** {inputs_str}
        - **Model:** GPBoost + TLFIG-Transformer
        - **Explanation:** The model analyzed your past baskets and identified {len(st.session_state.preds)} items likely to appear next.
        """)
    

    items_preview = ", ".join(st.session_state.confirmed_items[:5]) if st.session_state.confirmed_items else "N/A"
    top_item = st.session_state.preds[0]["item"]
    top_prob = st.session_state.preds[0]["probability"] * 100
    st.markdown(f'<div class="info-box">💡 <b>Insight:</b> Based on <b>{items_preview}</b>, top prediction: <b>{top_item}</b> ({top_prob:.1f}%)</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-header"><span class="step-num">6</span>Summary & Export</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(st.session_state.preds)}</div><div class="stat-label">Predictions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{top_prob:.1f}%</div><div class="stat-label">Top Confidence</div></div>', unsafe_allow_html=True)
    with col3:
        avg = np.mean([p["probability"] for p in st.session_state.preds]) * 100
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg:.1f}%</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            filepath = generate_pdf(st.session_state.domain, st.session_state.confirmed_items, st.session_state.preds)
            with open(filepath, "rb") as f:
                st.download_button("⬇️ Download PDF", f.read(), file_name=os.path.basename(filepath), mime="application/pdf", use_container_width=True)
    with exp_col2:
        csv = pd.DataFrame(st.session_state.preds).to_csv(index=False).encode()
        st.download_button("📊 Download CSV", csv, file_name="predictions.csv", mime="text/csv", use_container_width=True)
    with exp_col3:
        text_preds = "\n".join([f"{i+1}. {p['item']} ({p['probability']*100:.1f}%)" for i, p in enumerate(st.session_state.preds)])
        st.download_button("📋 Copy as Text", text_preds, file_name="predictions.txt", mime="text/plain", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.domain = None
        st.session_state.input_mode = None
        st.session_state.extracted_items = None
        st.session_state.confirmed_items = None
        st.session_state.preds = None
        st.session_state.manual_selection = []
        st.rerun()

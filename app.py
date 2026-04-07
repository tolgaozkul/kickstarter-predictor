import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Sayfa ayarları ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="centered"
)

# ── Dosyaları yükle (cache ile hızlı) ─────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("kickstarter_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

@st.cache_data
def load_lookup():
    with open("te_values.json")   as f: te      = json.load(f)
    with open("cat_medians.json") as f: cat_med = json.load(f)
    with open("ctr_medians.json") as f: ctr_med = json.load(f)
    with open("categories.json")  as f: cats    = json.load(f)
    with open("feature_list.json") as f: feats  = json.load(f)
    return te, cat_med, ctr_med, cats, feats

model, scaler, encoders = load_model()
te, cat_med, ctr_med, cats, feats = load_lookup()

CATEGORIES = sorted(cats["categories"])
COUNTRIES  = sorted(cats["countries"])

# ── Başlık ─────────────────────────────────────────────────────
st.markdown("# 🚀 Kickstarter Success Predictor")
st.markdown("**Will your campaign get funded?** Fill in the details below and find out.")
st.divider()

# ── Sol/Sağ kolon layout ───────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Campaign Details")

    category = st.selectbox(
        "Main Category",
        options=CATEGORIES,
        index=CATEGORIES.index("Dance") if "Dance" in CATEGORIES else 0,
        help="Choose the most relevant category for your project"
    )

    goal = st.number_input(
        "Funding Goal (USD $)",
        min_value=100,
        max_value=1_000_000,
        value=5_000,
        step=500,
        help="How much money do you need to raise?"
    )

    campaign_days = st.slider(
        "Campaign Duration (days)",
        min_value=7,
        max_value=60,
        value=30,
        help="Kickstarter allows 1–60 days. 30 days is the most common."
    )

with col2:
    st.markdown("### 🌍 Launch Details")

    country = st.selectbox(
        "Launch Country",
        options=COUNTRIES,
        index=COUNTRIES.index("US") if "US" in COUNTRIES else 0
    )

    launch_day = st.radio(
        "Launch Day",
        options=["Weekday (Mon–Fri)", "Weekend (Sat–Sun)"],
        index=0,
        help="Weekday launches perform slightly better."
    )

    launch_month = st.selectbox(
        "Launch Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ][x-1],
        index=2  # March default
    )

st.divider()

# ── Ek seçenekler (expander içinde, sade görünsün) ─────────────
with st.expander("⚙️ Advanced Options"):
    project_name = st.text_input(
        "Project Name (optional)",
        value="My Kickstarter Project",
        help="We'll use the name length and structure as signals."
    )
    round_goal = st.checkbox(
        "Round number goal? (e.g. $10,000 or $50,000)",
        value=False,
        help="Round goals are slightly associated with lower success rates."
    )

# ── Predict butonu ─────────────────────────────────────────────
predict_btn = st.button("🎯 Predict My Campaign", use_container_width=True, type="primary")

if predict_btn:
    # ── Feature'ları hesapla ───────────────────────────────────
    is_weekend   = 1 if "Weekend" in launch_day else 0
    name_length  = len(project_name)
    word_count   = len(project_name.split())
    name_has_num = int(any(c.isdigit() for c in project_name))
    name_has_exc = int('!' in project_name)
    launch_year  = 2026
    launch_dow   = 5 if is_weekend else 1
    launch_qtr   = (launch_month - 1) // 3 + 1
    is_early     = 0
    cur_is_usd   = 1 if country == "US" else 0

    log_goal     = np.log1p(goal)
    goal_round   = int(round_goal)

    # Target encoding
    te_cat = te["main_category"].get(category, 0.40)
    te_ctr = te["country"].get(country, 0.40)

    # Relative goal
    cat_median_val = cat_med.get(category, 5000)
    ctr_median_val = ctr_med.get(country, 5000)
    log_goal_vs_cat = np.log1p(goal / max(cat_median_val, 1))
    log_goal_vs_ctr = np.log1p(goal / max(ctr_median_val, 1))

    # Label encoding
    le_cat = encoders["main_category"].transform(
        [category] if category in encoders["main_category"].classes_
        else [encoders["main_category"].classes_[0]]
    )[0]
    le_ctr = encoders["country"].transform(
        [country] if country in encoders["country"].classes_
        else [encoders["country"].classes_[0]]
    )[0]
    # currency: US → USD, others → GBP as fallback
    currency_str = "USD" if country == "US" else "GBP"
    le_cur = encoders["currency"].transform(
        [currency_str] if currency_str in encoders["currency"].classes_
        else [encoders["currency"].classes_[0]]
    )[0]

    # ── DataFrame oluştur ──────────────────────────────────────
    input_data = pd.DataFrame([{
        'log_goal'         : log_goal,
        'campaign_days'    : campaign_days,
        'name_length'      : name_length,
        'word_count'       : word_count,
        'launch_month'     : launch_month,
        'launch_year'      : launch_year,
        'launch_dow'       : launch_dow,
        'launch_quarter'   : launch_qtr,
        'is_weekend_launch': is_weekend,
        'currency_is_usd'  : cur_is_usd,
        'goal_round_number': goal_round,
        'name_has_number'  : name_has_num,
        'name_has_exclaim' : name_has_exc,
        'is_early_period'  : is_early,
        'te_main_category' : te_cat,
        'te_country'       : te_ctr,
        'log_goal_vs_cat'  : log_goal_vs_cat,
        'log_goal_vs_ctr'  : log_goal_vs_ctr,
        'le_main_category' : le_cat,
        'le_country'       : le_ctr,
        'le_currency'      : le_cur,
    }])

    # Feature sırası önemli
    input_data = input_data[feats]

    # ── Tahmin ────────────────────────────────────────────────
    prob       = model.predict_proba(input_data)[0][1]
    prediction = "Successful" if prob >= 0.5 else "Failed"

    # ── Sonuç göster ──────────────────────────────────────────
    st.divider()
    st.markdown("## 📊 Prediction Result")

    if prediction == "Successful":
        st.success(f"### ✅ Likely to be FUNDED")
        color = "green"
    else:
        st.error(f"### ❌ Likely to FAIL")
        color = "red"

    # Büyük olasılık skoru
    st.markdown(
        f"<h1 style='text-align:center; color:{color}; font-size:72px;'>"
        f"{prob*100:.1f}%</h1>"
        f"<p style='text-align:center; color:gray;'>Predicted success probability</p>",
        unsafe_allow_html=True
    )

    # Progress bar
    st.progress(float(prob))

    # ── Detaylı insights ──────────────────────────────────────
    st.divider()
    st.markdown("### 💡 Key Signals for Your Campaign")

    c1, c2, c3 = st.columns(3)

    # Kategori başarı oranı
    cat_avg = te["main_category"].get(category, 0.40)
    overall_avg = 0.404
    cat_delta = (cat_avg - overall_avg) * 100
    with c1:
        st.metric(
            label=f"Category Avg ({category})",
            value=f"{cat_avg*100:.1f}%",
            delta=f"{cat_delta:+.1f}pp vs overall avg"
        )

    # Goal bracket
    if goal < 1000:
        bracket = "Micro (<$1K) — 53.8% avg"
        g_color = "normal"
    elif goal < 10000:
        bracket = "Small ($1K–10K) — 44.1% avg"
        g_color = "normal"
    elif goal < 50000:
        bracket = "Medium ($10–50K) — 31.0% avg"
        g_color = "off"
    elif goal < 100000:
        bracket = "Large ($50–100K) — 17.2% avg"
        g_color = "inverse"
    else:
        bracket = "XL (>$100K) — 7.5% avg"
        g_color = "inverse"
    with c2:
        st.metric(label="Goal Bracket", value=f"${goal:,}", delta=bracket, delta_color=g_color)

    # Weekend vs weekday
    with c3:
        launch_signal = "Weekend (-2pp)" if is_weekend else "Weekday (+2pp)"
        st.metric(
            label="Launch Day",
            value="Weekend" if is_weekend else "Weekday",
            delta=launch_signal,
            delta_color="inverse" if is_weekend else "normal"
        )

    # ── Tips ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎯 Tips to Improve Your Odds")

    tips = []
    if goal > 10000:
        tips.append(f"💰 **Lower your goal.** At ${goal:,}, you're in a bracket with lower success rates. Consider reducing to under $10,000.")
    if is_weekend:
        tips.append("📅 **Switch to a weekday launch.** Weekday campaigns perform ~2pp better on average.")
    if round_goal:
        tips.append("🔢 **Avoid round number goals.** Specific goals like $9,500 signal more careful cost planning than $10,000.")
    if cat_avg < overall_avg:
        tips.append(f"🎭 **Consider a different category.** {category} has a below-average success rate ({cat_avg*100:.1f}%). Dance, Theater, or Comics perform significantly better.")
    if not tips:
        tips.append("🌟 **Your campaign looks well-positioned!** Keep the momentum and promote heavily in the first 24–48 hours — early traction is critical.")

    for tip in tips:
        st.markdown(f"- {tip}")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: XGBoost (Optuna-optimized) | AUC: 0.743 | "
    "Trained on 331,462 Kickstarter campaigns (2009–2018) | "
    "For demo purposes only"
)

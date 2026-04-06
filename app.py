import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mobile App Success Predictor",
    page_icon="📱",
    layout="wide",
)

# ─── Custom CSS (Dark Theme) ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1c2a1c, #162616);
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-rating {
        font-size: 3.5rem;
        font-weight: 900;
        color: #3fb950;
        line-height: 1.1;
    }

    .result-label {
        color: #8b949e;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    .star-row {
        font-size: 1.5rem;
        margin-top: 0.4rem;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        color: #8b949e;
        margin: 0.2rem;
    }

    /* Instruction box */
    .info-box {
        background: #0d2137;
        border: 1px solid #1f6feb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #79c0ff;
    }

    /* Predict button override */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #238636, #196127) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Table styling */
    .rec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .rec-table th {
        background: #21262d;
        color: #58a6ff;
        padding: 0.6rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid #30363d;
    }
    .rec-table td {
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid #21262d;
        color: #e6edf3;
    }
    .rec-table tr:hover td {
        background: #161b22;
    }

    /* Divider */
    hr {
        border-color: #21262d;
    }

    /* Selectbox / number input labels */
    label {
        color: #c9d1d9 !important;
        font-size: 0.9rem !important;
    }

    /* Hide default streamlit menu */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: Build/Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_data_and_models():
    """
    Loads the Google Play Store CSV, cleans it, trains the model
    and builds the recommendation matrix – all in memory.
    Falls back to demo data if the CSV is absent.
    """
    # ── Try loading real data ──
    csv_candidates = [
        "googleplaystore.csv",
        os.path.join(os.path.dirname(__file__), "googleplaystore.csv"),
    ]
    df = None
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        # ── Synthetic demo data so the app still works without the CSV ──
        np.random.seed(42)
        n = 500
        cats = ["GAME", "PRODUCTIVITY", "TOOLS", "SOCIAL", "COMMUNICATION",
                "ENTERTAINMENT", "SHOPPING", "FINANCE", "HEALTH_AND_FITNESS", "EDUCATION"]
        content_ratings = ["Everyone", "Teen", "Everyone 10+", "Mature 17+", "Adults only 18+"]
        df = pd.DataFrame({
            "App": [f"SampleApp_{i}" for i in range(n)],
            "Category": np.random.choice(cats, n),
            "Rating": np.clip(np.random.normal(4.2, 0.5, n), 1, 5),
            "Reviews": np.random.randint(10, 500000, n).astype(str),
            "Size": [f"{np.random.randint(1,100)}M" for _ in range(n)],
            "Installs": [f"{np.random.choice([1000,10000,100000,1000000,10000000])}+" for _ in range(n)],
            "Type": np.random.choice(["Free", "Paid"], n, p=[0.9, 0.1]),
            "Price": ["0" if t == "Free" else f"{np.random.uniform(0.99,9.99):.2f}"
                      for t in np.random.choice(["Free", "Paid"], n, p=[0.9, 0.1])],
            "Content Rating": np.random.choice(content_ratings, n),
            "Genres": np.random.choice(cats, n),
            "Current Ver": ["1.0"] * n,
            "Android Ver": ["5.0"] * n,
        })

    # ── Cleaning ──
    df.drop_duplicates(inplace=True)
    df["Installs"] = pd.to_numeric(df["Installs"].astype(str).str.replace("[+,]", "", regex=True), errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"].astype(str).str.replace("[+,]", "", regex=True), errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace("$", "", regex=False), errors="coerce")

    def convert_size(size):
        s = str(size)
        if 'M' in s:
            return float(s.replace('M', ''))
        elif 'k' in s:
            return float(s.replace('k', '')) / 1024
        return np.nan

    df["Size"] = df["Size"].apply(convert_size)
    df.replace("Varies with device", np.nan, inplace=True)
    df.dropna(subset=["Rating"], inplace=True)
    df.ffill(inplace=True)

    # IQR filter on Rating
    Q1, Q3 = df["Rating"].quantile(0.25), df["Rating"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Rating"] >= Q1 - 1.5 * IQR) & (df["Rating"] <= Q3 + 1.5 * IQR)]

    # Encoding
    df["Type_enc"] = df["Type"].map({"Free": 0, "Paid": 1}).fillna(0).astype(int)

    le_cat = LabelEncoder()
    le_con = LabelEncoder()
    df["Category_enc"] = le_cat.fit_transform(df["Category"].astype(str))
    df["Content_enc"] = le_con.fit_transform(df["Content Rating"].astype(str))

    # ── Train XGBoost ──
    features = ["Category_enc", "Reviews", "Size", "Installs", "Type_enc", "Content_enc"]
    df_model = df.dropna(subset=features + ["Rating"])
    X = df_model[features].fillna(0)
    y = df_model["Rating"]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                         max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=42)
    model.fit(X, y)

    # ── Recommendation Matrix ──
    rec_df = df_model[["App", "Category_enc", "Installs", "Type_enc", "Reviews"]].copy().reset_index(drop=True)
    scaler = StandardScaler()
    rec_scaled = scaler.fit_transform(rec_df[["Category_enc", "Installs", "Type_enc", "Reviews"]])
    sim_matrix = cosine_similarity(rec_scaled)

    return model, le_cat, le_con, df, df_model.reset_index(drop=True), rec_df, sim_matrix


def recommend_similar(cat_enc, installs, price_type, reviews,
                       df_model, rec_df, sim_matrix, top_n=5):
    """Return similar apps based on a synthetic 'new app' feature vector."""
    query = np.array([[cat_enc, installs, price_type, reviews]], dtype=float)
    scaler_q = StandardScaler()
    rec_vals = rec_df[["Category_enc", "Installs", "Type_enc", "Reviews"]].values
    scaler_q.fit(rec_vals)
    q_scaled = scaler_q.transform(query)
    all_scaled = scaler_q.transform(rec_vals)
    sims = cosine_similarity(q_scaled, all_scaled)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]

    rows = []
    for i in top_idx:
        row = df_model.iloc[i]
        rows.append({
            "App Name": row.get("App", "—"),
            "Category": row.get("Category", "—"),
            "Rating": f"{row.get('Rating', 0):.1f} ⭐",
            "Installs": f"{int(row.get('Installs', 0)):,}",
            "Type": "Free" if row.get("Type_enc", 0) == 0 else "Paid",
            "Similarity": f"{sims[i]*100:.1f}%",
        })
    return pd.DataFrame(rows)


def rating_stars(rating):
    full = int(round(rating))
    return "⭐" * full + "☆" * (5 - full)


def success_tier(rating):
    if rating >= 4.5:
        return "🏆 Top Performer", "#3fb950"
    elif rating >= 4.0:
        return "✅ Strong Performer", "#58a6ff"
    elif rating >= 3.5:
        return "⚡ Average Performer", "#d29922"
    else:
        return "⚠️ Needs Improvement", "#f85149"


# ─── Load Everything ──────────────────────────────────────────────────────────
with st.spinner("Initialising prediction engine…"):
    model, le_cat, le_con, df_raw, df_model, rec_df, sim_matrix = load_data_and_models()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📱 Mobile App Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered rating prediction & competitive intelligence for app developers</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
💡 <strong>How to use this tool:</strong> Fill in your app's specifications in the form below
and click <em>Predict My App's Success</em>. The model will estimate your app's expected rating
on the Google Play Store and show you which existing apps you'll be competing with.
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Layout: 2 columns ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown("### 🛠️ App Specifications")

    category_options = sorted(le_cat.classes_.tolist())
    content_options = sorted(le_con.classes_.tolist())

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("📂 App Category", category_options,
                                 help="Select the primary category of your app")
    with col2:
        content_rating = st.selectbox("🎯 Content Rating", content_options,
                                       help="Target audience age group")

    col3, col4 = st.columns(2)
    with col3:
        price_type = st.radio("💰 Price Model", ["Free", "Paid"],
                               horizontal=True,
                               help="Is your app free or paid?")
    with col4:
        size = st.number_input("📦 App Size (MB)", min_value=0.1, max_value=500.0,
                                value=20.0, step=0.5,
                                help="App size in megabytes")

    col5, col6 = st.columns(2)
    with col5:
        installs = st.number_input("📥 Expected Installs", min_value=0,
                                    max_value=1_000_000_000, value=10_000,
                                    step=1_000,
                                    help="Expected number of installs")
    with col6:
        reviews = st.number_input("💬 Expected Reviews", min_value=0,
                                   max_value=10_000_000, value=500,
                                   step=100,
                                   help="Expected number of user reviews")

    st.markdown("")
    predict_btn = st.button("🚀 Predict My App's Success", use_container_width=True)

    # Metadata badges
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <span class="badge">📊 {len(df_raw):,} apps in training data</span>
    <span class="badge">🤖 XGBoost Regressor</span>
    <span class="badge">📈 6 features</span>
    """, unsafe_allow_html=True)


# ─── Prediction & Recommendations ─────────────────────────────────────────────
with right_col:
    st.markdown("### 📊 Prediction Results")

    if predict_btn:
        # Encode inputs
        cat_enc = le_cat.transform([category])[0]
        con_enc = le_con.transform([content_rating])[0]
        type_enc = 0 if price_type == "Free" else 1

        X_input = np.array([[cat_enc, reviews, size, installs, type_enc, con_enc]])
        predicted_rating = float(model.predict(X_input)[0])
        predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

        tier_label, tier_color = success_tier(predicted_rating)

        # Rating card
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Expected Rating</div>
            <div class="result-rating">{predicted_rating:.1f}<span style="font-size:1.5rem;color:#8b949e">/5.0</span></div>
            <div class="star-row">{rating_stars(predicted_rating)}</div>
            <div style="margin-top:0.8rem;">
                <span style="background:{tier_color}22;border:1px solid {tier_color};
                      border-radius:20px;padding:0.25rem 0.9rem;
                      color:{tier_color};font-weight:600;font-size:0.9rem;">
                    {tier_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Category", category.replace("_", " ").title())
        m2.metric("Price Model", price_type)
        m3.metric("Content Rating", content_rating)

        st.divider()

        # Recommendations
        st.markdown("#### 🏆 Similar Apps You Might Compete With")
        rec_table = recommend_similar(cat_enc, installs, type_enc, reviews,
                                       df_model, rec_df, sim_matrix)

        st.markdown(rec_table.to_html(index=False, classes="rec-table"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("💡 Similarity is computed using cosine similarity on category, installs, price type, and reviews.")

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#484f58;">
            <div style="font-size:4rem;margin-bottom:1rem;">🔮</div>
            <div style="font-size:1.1rem;color:#6e7681;font-weight:600;">
                Fill in your app specifications<br>and click Predict to see results
            </div>
            <div style="margin-top:1.5rem;font-size:0.85rem;color:#484f58;max-width:300px;margin-left:auto;margin-right:auto;">
                The model will estimate your expected rating and surface similar competitive apps in the market.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.78rem;padding:0.5rem 0 1rem;">
    Built with XGBoost · Google Play Store Dataset · Streamlit
</div>
""", unsafe_allow_html=True)

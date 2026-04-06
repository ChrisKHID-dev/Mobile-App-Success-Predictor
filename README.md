# 📱 Mobile App Success Predictor & Recommender

An AI-powered Streamlit app that predicts your app's expected Play Store rating
and surfaces similar competitive apps.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your dataset
Place **`googleplaystore.csv`** in the same folder as `app.py`.
If the CSV is absent the app runs on built-in synthetic demo data automatically.

### 3. Run
```bash
streamlit run app.py
```

---

## 📋 Inputs

| Field | Description |
|---|---|
| App Category | Primary Play Store category (e.g. GAME, TOOLS) |
| App Size (MB) | Size of the APK in megabytes |
| Price Model | Free or Paid |
| Expected Installs | Projected install count |
| Content Rating | Target audience (Everyone, Teen, …) |
| Expected Reviews | Projected review count |

## 📊 Outputs

- **Predicted Rating** — Random Forest Regression prediction with star display and tier badge
- **Competition Table** — Top 5 similar apps via cosine similarity on category, installs, price, and reviews

---

## 🛠 Tech Stack

- **Model** : Random Forest Regressor (trained on Google Play Store dataset)
- **Recommendation** : Cosine Similarity (scikit-learn)
- **UI** : Streamlit with custom dark-theme CSS

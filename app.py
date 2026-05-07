import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import streamlit.components.v1 as components

# =========================
# LOAD DATA
# =========================
data = pd.read_csv(r"C:\Users\Colab\Downloads\project\small_data.csv")
data = data.reset_index(drop=True)
data['main_category'] = data['product_category_tree'].fillna("Unknown").apply(
    lambda x: str(x)
    .replace('[', '')
    .replace(']', '')
    .replace('"', '')
    .split(">>")[0]
    .strip()
    if ">>" in str(x) else "Unknown"
)
# =========================
# LOAD MODEL + EMBEDDINGS
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(product_name):
    product_name = product_name.strip().lower()

    if not product_name:
        return None

    query_embedding = model.encode([product_name])
    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

    indices = sim_scores.argsort()[::-1][1:20]

    return data.iloc[indices][[
        'product_name',
        'image',
        'brand',
        'retail_price',
        'discounted_price',
        'product_rating'
    ]].drop_duplicates().head(6)

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Product Recommender", layout="centered")

# Title
st.markdown("<h1 style='text-align:center;'>🛒 AI Product Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:gray;'>Find similar products instantly using AI</p>", unsafe_allow_html=True)

st.divider()

# Input
user_input = st.text_input(
    "🔍 Search Product",
    placeholder="e.g., sofa, shoes, kurti",
    key="search_box"
)
category = st.selectbox(
    "📂 Select Category",
    ["All"] + sorted(data['main_category'].unique().tolist())
)
st.info("💡 Select a category and then search a related product (e.g., Clothing → jeans)")
# Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_clicked = st.button("🚀 Recommend", key="recommend_button")

st.divider()

# Results
if search_clicked:

    if user_input.strip() == "":
        st.warning("⚠ Please enter a product name")

    else:
        # ✅ FILTER FIRST
        if category != "All":
            filtered_data = data[data['main_category'] == category]
        else:
            filtered_data = data

        # ✅ THEN RECOMMEND
        results = recommend(user_input)

        # ✅ KEEP ONLY FILTERED RESULTS
        results = results[
            results['product_name'].isin(filtered_data['product_name'])
        ]

        st.subheader("✨ Recommended Products")

        cols = st.columns(3)

        for i, row in results.iterrows():
            with cols[i % 3]:

                # Fix image
                try:
                    image_url = json.loads(row['image'])[0]
                except:
                    image_url = row['image']

                st.image(image_url, use_container_width=True)

                # HTML CARD
                components.html(
                    f"""
                    <div style="
                        padding:12px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        margin-bottom:12px;
                        background-color:#ffffff;
                        text-align:center;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                        font-family:sans-serif;
                    ">
                        <b style="font-size:14px;">{row['product_name']}</b><br>

                        <span style="color:gray;font-size:12px;">
                            Brand: {row['brand']}
                        </span><br>

                        <span style="color:green;font-size:16px;">
                            ₹{row['discounted_price']}
                        </span>

                        <span style="text-decoration:line-through;color:red;font-size:12px;">
                            ₹{row['retail_price']}
                        </span><br>

                        <span style="color:orange;">
                            ⭐ {row['product_rating']}
                        </span>
                    </div>
                    """,
                    height=220
                )
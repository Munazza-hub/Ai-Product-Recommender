import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit.components.v1 as components

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("small_data.csv")

data = load_data()
data = data.reset_index(drop=True)

# =========================
# LOAD MODEL + EMBEDDINGS
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

embeddings = load_embeddings()

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(product_name):

    product_name = product_name.strip().lower()

    if not product_name:
        return None

    # Encode query
    query_embedding = model.encode([product_name])

    # Similarity scores
    sim_scores = cosine_similarity(
        query_embedding,
        embeddings
    ).flatten()

    # Get sorted indices
    indices = sim_scores.argsort()[::-1]

    # Keep only VALID indices
    indices = [i for i in indices if i < len(data)]

    # Top recommendations
    indices = indices[1:20]

    # Return products
    return data.iloc[indices][[
        'name',
        'img',
        'seller',
        'price',
        'mrp',
        'rating'
    ]].drop_duplicates().head(6)
# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="AI Product Recommender",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🛒 AI Product Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Find similar products instantly using AI</p>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# INPUT
# =========================
user_input = st.text_input(
    "🔍 Search Product",
    placeholder="e.g., shoes, kurti, hoodie",
    key="search_box"
)

# =========================
# BUTTON
# =========================
col1, col2, col3 = st.columns([1,1,1])

with col2:
    search_clicked = st.button(
        "🚀 Recommend",
        key="recommend_button"
    )

st.divider()

# =========================
# RESULTS
# =========================
if search_clicked:

    if user_input.strip() == "":
        st.warning("⚠ Please enter a product name")

    else:

        results = recommend(user_input)

        st.subheader("✨ Recommended Products")

        cols = st.columns(3)

        for i, row in results.iterrows():

            with cols[i % 3]:

                # =========================
                # IMAGE
                # =========================
                try:
                    image_url = str(row['img']).split(";")[0]

                    image_url = image_url.replace(
                        "http://",
                        "https://"
                    )

                    st.image(
                        image_url,
                        use_container_width=True
                    )

                except:
                    st.image(
                        "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",
                        use_container_width=True
                    )

                # =========================
                # CARD
                # =========================
                components.html(
                    f"""
                    <div style="
                        padding:12px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        margin-bottom:12px;
                        background-color:#ffffff;
                        text-align:center;
                        box-shadow:2px 2px 10px rgba(0,0,0,0.05);
                        font-family:sans-serif;
                    ">

                        <b style="font-size:14px;">
                            {row['name']}
                        </b>

                        <br><br>

                        <span style="
                            color:gray;
                            font-size:12px;
                        ">
                            Brand: {row['seller']}
                        </span>

                        <br><br>

                        <span style="
                            color:green;
                            font-size:16px;
                        ">
                            ₹{row['price']}
                        </span>

                        <span style="
                            text-decoration:line-through;
                            color:red;
                            font-size:12px;
                        ">
                            ₹{row['mrp']}
                        </span>

                        <br><br>

                        <span style="
                            color:orange;
                        ">
                            ⭐ {row['rating']}
                        </span>

                    </div>
                    """,
                    height=250
                )
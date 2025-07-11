import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np
from collections import defaultdict

# ----------------------
# Load data (real + synthetic)
# ----------------------

@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    ratings = ratings.drop(columns=["timestamp"])

    # Load movies
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children’s", "Comedy", "Crime", "Documentary",
        "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]
    movie_cols = ["item_id", "title", "release_date", "video_release", "IMDb_URL"] + genre_cols
    movies_df = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, names=movie_cols)
    movies_df["genres"] = movies_df.apply(lambda row: ", ".join([g for g in genre_cols if row[g] == 1]), axis=1)
    movies_df = movies_df[["item_id", "title", "genres"]]

    return ratings, movies_df

# Load models (already trained)
@st.cache_resource
def train_models(real_ratings, synthetic_ratings):
    reader = Reader(rating_scale=(1, 5))
    real_dataset = Dataset.load_from_df(real_ratings, reader)
    combined_df = pd.concat([real_ratings, synthetic_ratings])

    combined_dataset = Dataset.load_from_df(combined_df, reader)

    train_real, _ = train_test_split(real_dataset, test_size=0.2, random_state=42)
    train_comb, _ = train_test_split(combined_dataset, test_size=0.2, random_state=42)

    algo_real = SVD()
    algo_real.fit(train_real)

    algo_comb = SVD()
    algo_comb.fit(train_comb)

    return algo_real, algo_comb, train_real, train_comb

# ---------------------
# Recommendation Logic
# ---------------------

def get_top_10_recommendations(user_id, algo, movies_df, trainset, genre_filter=None):
    try:
        rated_items = set([
            iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]
        ])
    except:
        rated_items = set()

    all_items = set(trainset._raw2inner_id_items.keys())
    unrated_items = all_items - rated_items

    predictions = [algo.predict(user_id, iid) for iid in unrated_items]
    top_10 = sorted(predictions, key=lambda x: x.est, reverse=True)

    top_10_df = pd.DataFrame([(pred.iid, pred.est) for pred in top_10], columns=["item_id", "pred_rating"])
    top_10_df = top_10_df.merge(movies_df, on="item_id", how="left")

    # Apply genre filter if selected
    if genre_filter and genre_filter != "All":
        top_10_df = top_10_df[top_10_df["genres"].str.contains(genre_filter, case=False)]

    return top_10_df[["title", "genres", "pred_rating"]].head(10)

def plot_bar_chart(df, model_type):
    fig, ax = plt.subplots()
    ax.barh(df["title"], df["pred_rating"], color="skyblue")
    ax.set_xlabel("Predicted Rating")
    ax.set_title(f"Top-10 Recommendations ({model_type})")
    ax.invert_yaxis()
    st.pyplot(fig)

# ----------------------
# Streamlit App
# ----------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System (Real vs Synthetic Data)")

ratings, movies_df = load_data()

# For demo, you may load synthetic data from earlier step or saved file
synthetic_df = pd.read_csv("synthetic_ratings.csv")  # ensure this file exists

# Train models
algo_real, algo_comb, train_real, train_comb = train_models(ratings, synthetic_df)

# Sidebar inputs
user_id = st.sidebar.number_input("Enter User ID (1–943)", min_value=1, max_value=943, value=1)
model_type = st.sidebar.radio("Select Model", ["Real Only", "Real + Synthetic"])
genre_filter = st.sidebar.selectbox("Filter by Genre", ["All"] + sorted(list(set(", ".join(movies_df["genres"]).split(", ")))))

# Recommendation
st.subheader(f"Top-10 Recommendations for User {user_id}")

model = algo_real if model_type == "Real Only" else algo_comb
trainset = train_real if model_type == "Real Only" else train_comb

top_10 = get_top_10_recommendations(str(user_id), model, movies_df, trainset, genre_filter)

st.dataframe(top_10)

plot_bar_chart(top_10, model_type)

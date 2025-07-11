import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Load Data
@st.cache_data
def load_data():
    real = pd.read_csv("recommendations_real.csv")
    combined = pd.read_csv("recommendations_combined.csv")
    return real, combined

real_df, combined_df = load_data()

# 2. Sidebar Controls
st.sidebar.title("🔍 Recommender Controls")
user_id = st.sidebar.number_input("Enter User ID (1–943)", min_value=1, max_value=943, value=1)
model_choice = st.sidebar.radio("Select Model:", ["Real Only", "Real + Synthetic"])
genre_filter = st.sidebar.selectbox(
    "Filter by Genre:",
    ["All"] + sorted(list(set(", ".join(real_df["genres"]).split(", "))))
)

# 3. Filter Based on Selection
df = real_df if model_choice == "Real Only" else combined_df
user_df = df[df["user_id"] == user_id]

# Apply genre filter
if genre_filter != "All":
    user_df = user_df[user_df["genres"].str.contains(genre_filter, case=False)]

top_10 = user_df.sort_values("pred_rating", ascending=False).head(10)

# 4. Display Results
st.title("🎬 Movie Recommender System")
st.subheader(f"Top 10 Recommendations for User {user_id} ({model_choice})")

st.dataframe(top_10[["title", "genres", "pred_rating"]].reset_index(drop=True))

# 5. Plot Results
if not top_10.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_10["title"], top_10["pred_rating"], color="skyblue")
    ax.set_xlabel("Predicted Rating")
    ax.set_title("Recommended Movies")
    ax.invert_yaxis()
    st.pyplot(fig)
else:
    st.warning("⚠️ No recommendations match the selected genre.")

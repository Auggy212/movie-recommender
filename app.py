import pandas as pd
import streamlit as st
import plotly.express as px

# Set wide mode and page config
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")

# Header
st.markdown("""
    <style>
    .main-title {
        font-size:36px;
        font-weight:bold;
        color:#FF4B4B;
    }
    .subtext {
        font-size:16px;
        color:#333;
    }
    </style>
    <div class='main-title'>🎬 AI Movie Recommender Dashboard</div>
    <div class='subtext'>Compare recommendations from Real Data vs Real + Synthetic Data using genre filters and rating insights</div>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.title("🔧 Filters")
user_id = st.sidebar.number_input("Select User ID (1 - 943)", min_value=1, max_value=943, value=1)
model_choice = st.sidebar.radio("Select Model", ["Real Only", "Real + Synthetic"])
genre_filter = st.sidebar.selectbox("Select Genre", ["All", "Action", "Adventure", "Animation", "Children’s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

# Load CSVs
@st.cache_data
def load_data():
    real = pd.read_csv("recommendations_real.csv")
    combined = pd.read_csv("recommendations_combined.csv")
    return real, combined

real_df, combined_df = load_data()

# Filter data
df = real_df if model_choice == "Real Only" else combined_df
user_df = df[df["user_id"] == user_id]

# Filter by genre
if genre_filter != "All":
    user_df = user_df[user_df["genres"].str.contains(genre_filter, case=False, na=False)]

# Top 10 Recommendations
top_10 = user_df.sort_values("pred_rating", ascending=False).head(10)

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"🎯 Top 10 Recommendations for User {user_id} ({model_choice})")
    st.dataframe(top_10[["title", "genres", "pred_rating"]].reset_index(drop=True), use_container_width=True)

with col2:
    if not top_10.empty:
        fig = px.bar(top_10, y="title", x="pred_rating", orientation='h',
                     color_discrete_sequence=["#FF6361"], labels={'pred_rating': 'Predicted Rating'}, height=400)
        fig.update_layout(title="📊 Predicted Ratings", yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No recommendations available for the selected genre.")

# Footer Metrics
with st.expander("📈 View Metrics & Summary"):
    summary = {
        "Total Movies Recommended": len(top_10),
        "Genre Applied": genre_filter,
        "Model Used": model_choice
    }
    st.json(summary)
    if not top_10.empty:
        avg_rating = round(top_10["pred_rating"].mean(), 2)
        st.success(f"📌 Average Predicted Rating: {avg_rating}")
    else:
        st.info("📌 No average available — empty recommendation list.")

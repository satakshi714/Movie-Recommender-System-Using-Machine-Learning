import streamlit as st
import pickle
import requests
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="CineMatch", layout="wide")

# -----------------------------
# API KEY (CLOUD SAFE)
# -----------------------------
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = ""   # fallback for local

# -----------------------------
# LOAD / GENERATE MODEL (FIXED)
# -----------------------------
try:
    if not os.path.exists("artifacts/movie_list.pkl"):

        st.warning("Generating model... please wait ⏳")

        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Load CSV
        movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
        credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

        # Merge
        movies_df = movies_df.merge(credits_df, on="title")

        # Minimal preprocessing
        movies_df = movies_df[['movie_id', 'title', 'overview']]
        movies_df['overview'] = movies_df['overview'].fillna('')

        # Vectorization
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies_df['overview']).toarray()

        similarity = cosine_similarity(vectors)

        # Save artifacts
        os.makedirs("artifacts", exist_ok=True)
        pickle.dump(movies_df, open('artifacts/movie_list.pkl', 'wb'))
        pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

    # Load model
    movies = pickle.load(open('artifacts/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))

except Exception as e:
    st.error(f"❌ Failed to load/generate model: {e}")
    st.stop()

movie_list = movies['title'].values

# -----------------------------
# SAFE API CALL
# -----------------------------
def safe_request(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

# -----------------------------
# FETCH MOVIE DATA
# -----------------------------
def fetch_movie_data(movie_id):
    if not API_KEY:
        return "https://via.placeholder.com/300x450", "N/A", "No API key"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = safe_request(url)

    poster = data.get('poster_path')
    rating = data.get('vote_average', "N/A")
    overview = data.get('overview', "No description available")

    poster_url = (
        "https://image.tmdb.org/t/p/w500/" + poster
        if poster else "https://via.placeholder.com/300x450"
    )

    return poster_url, rating, overview

# -----------------------------
# FETCH TRAILER
# -----------------------------
def fetch_trailer(movie_id):
    if not API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
    data = safe_request(url)

    for video in data.get('results', []):
        if video.get('type') == "Trailer":
            return "https://www.youtube.com/watch?v=" + video.get('key')

    return None

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_movies(query, movies):
    if not query:
        return list(movies[:20])

    query = query.lower()
    results = [m for m in movies if query in m.lower()]
    return results[:20] if results else list(movies[:20])

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
    except:
        return [], [], [], [], []

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names, posters, ratings, overviews, trailers = [], [], [], [], []

    for i in distances[1:6]:
        try:
            movie_id = movies.iloc[i[0]].movie_id
            names.append(movies.iloc[i[0]].title)

            poster, rating, overview = fetch_movie_data(movie_id)
            trailer = fetch_trailer(movie_id)

            posters.append(poster)
            ratings.append(rating)
            overviews.append(overview)
            trailers.append(trailer)

        except:
            continue

    return names, posters, ratings, overviews, trailers

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎬 CineMatch</div>", unsafe_allow_html=True)

# -----------------------------
# SEARCH UI
# -----------------------------
search_query = st.text_input("🔍 Search Movie")
filtered_movies = search_movies(search_query, movie_list)

selected_movie = st.selectbox("Select Movie", filtered_movies)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("✨ Recommend Movies"):

    with st.spinner("Finding best movies... 🍿"):
        names, posters, ratings, overviews, trailers = recommend(selected_movie)

    if not names:
        st.warning("⚠️ No recommendations found")
    else:
        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.image(posters[i], use_container_width=True)
                st.write(names[i])
                st.write(f"⭐ {ratings[i]}")
                st.write(overviews[i][:80] + "...")

                if trailers[i]:
                    st.link_button("▶ Trailer", trailers[i])

                st.markdown("</div>", unsafe_allow_html=True)
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# =========================
# 🔹 LOAD & TRAIN MODEL
# =========================
df = pd.read_csv("data/movies.csv")

cv = CountVectorizer()
vectors = cv.fit_transform(df["genres"]).toarray()

similarity = cosine_similarity(vectors)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((df, similarity), f)

# =========================
# 🔹 FLASK API
# =========================
app = Flask(__name__)

with open("model.pkl", "rb") as f:
    movies, similarity = pickle.load(f)

def recommend(movie):
    if movie not in movies["title"].values:
        return None

    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]

    movie_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

@app.route("/")
def home():
    return "🎬 Movie Recommender Running"

@app.route("/recommend", methods=["POST"])
def get_rec():
    data = request.json

    if not data or "movie" not in data:
        return jsonify({"error": "Provide movie name"}), 400

    recs = recommend(data["movie"])

    if recs is None:
        return jsonify({"error": "Movie not found"}), 404

    return jsonify({"recommendations": recs})

# =========================
# 🔹 RUN (IMPORTANT FOR DOCKER)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Docker/GCP uses 8080
    app.run(host="0.0.0.0", port=port)

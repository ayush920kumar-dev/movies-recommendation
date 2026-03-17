# movies-recommendation
machine learning model designed to recommend movies based on preference.
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 🔹 LOAD & TRAIN MODEL
# =========================
df = pd.read_csv("data/movies.csv")

cv = CountVectorizer()
vectors = cv.fit_transform(df["genres"]).toarray()

similarity = cosine_similarity(vectors)

# Save model
pickle.dump((df, similarity), open("model.pkl", "wb"))

# =========================
# 🔹 FLASK API
# =========================
app = Flask(__name__)

movies, similarity = pickle.load(open("model.pkl", "rb"))

def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

@app.route("/")
def home():
    return "Movie Recommender Running 🎬"

@app.route("/recommend", methods=["POST"])
def get_rec():
    data = request.json
    movie = data["movie"]

    try:
        recs = recommend(movie)
        return jsonify({"recommendations": recs})
    except:
        return jsonify({"error": "Movie not found"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

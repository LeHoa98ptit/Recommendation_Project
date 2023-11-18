import json

from flask import Flask, request, jsonify
import pandas as pd
from surprise import dump
from flask import make_response

# load cleaned dataset
song_df = pd.read_csv("Data/cleaned_song_dataset.csv")

# Load the model
collab_model = dump.load("model/collab_model.pkl")[1]

# Flask app
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello"


# API endpoint for recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get user_id from query parameters
    user_id = str(request.args.get('user_id'))
    print(user_id)

    if user_id is None:
        return jsonify({"error": "Missing 'user_id' parameter"}), 400

    # Collaborative Filtering Recommendations
    recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)
    print(recommendations)

    return make_response(jsonify(recommendations), 200)


def collaborative_filtering_recommendation(user_id, model, df):
    # Get a list of all songs listened to by the user
    listened_songs = df[df['user'] == user_id]['title'].tolist()

    # Find unheard songs
    all_songs = df['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    # Give users suggestions from unheard songs
    recommendations = [(song, model.predict(user_id, song).est) for song in unheard_songs]

    # Sort and get top k suggestions
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return recommendations


if __name__ == '__main__':
    app.run(debug=True)


"""

# API endpoint for recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get user and listened songs from request
    user_id = request.json['user_id']
    listened_songs = request.json['listened_songs']

    # Collaborative Filtering Recommendations
    recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df, listened_songs)

    return jsonify(recommendations)


def collaborative_filtering_recommendation(user_id, model, df, listened_songs):
    # Find unheard songs
    all_songs = df['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    # Give users suggestions from unheard songs
    recommendations = [(song, model.predict(user_id, song).est) for song in unheard_songs]

    # Sort and get top k suggestions
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return recommendations


"""
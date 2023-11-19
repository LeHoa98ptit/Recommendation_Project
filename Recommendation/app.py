from flask import Flask, request, jsonify, render_template
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
def index():
    return render_template("index.html")


# API endpoint for recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = str(request.args.get('user_id'))

    if user_id is None:
        return jsonify({"error": "Missing 'user_id' parameter"}), 400

    listened_songs, recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)

    return make_response(jsonify({
        'user_songs': listened_songs,
        'recommendations': recommendations
    }))


def collaborative_filtering_recommendation(user_id, model, df):
    listened_songs = df[df['user'] == user_id]['title'].tolist()

    all_songs = df['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    recommendations = [(song, str(model.predict(user_id, song).est)) for song in unheard_songs]

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return listened_songs, recommendations


# Create an API endpoint to return a list of users
@app.route('/get_users')
def get_users():
    users = song_df['user'].unique().tolist()
    return jsonify(users)


# Create an API endpoint to return the user's playlist and recommendations
@app.route('/get_user_and_recommendations', methods=['POST'])
def get_user_and_recommendations():
    user_id = request.json['user_id']
    user_songs, recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)

    return jsonify({
        'user_songs': user_songs,
        'recommendations': recommendations
    })


if __name__ == '__main__':
    app.run(debug=True)

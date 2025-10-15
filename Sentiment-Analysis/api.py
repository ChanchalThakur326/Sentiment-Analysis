from flask import Flask, request, jsonify, render_template
import re
from io import BytesIO
import os
import traceback
import base64

# nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Matplotlib non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import pickle

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

# -------------------- ROUTES -------------------- #
@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load models
        model_path = "Models/model_xgb.pkl"
        scaler_path = "Models/scaler.pkl"
        cv_path = "Models/countVectorizer.pkl"

        if not all(os.path.exists(p) for p in [model_path, scaler_path, cv_path]):
            return jsonify({"error": "One or more model files are missing in Models/ folder"})

        predictor = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        cv = pickle.load(open(cv_path, "rb"))

        # Check input type: file or text
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            if "Sentence" not in data.columns:
                return jsonify({"error": "CSV file must contain a 'Sentence' column"})

            predictions_csv, graph_base64 = bulk_prediction(predictor, scaler, cv, data)

            return jsonify({
                "predictions_csv": base64.b64encode(predictions_csv.getvalue()).decode('utf-8'),
                "graph_base64": graph_base64
            })

        elif request.json and "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        else:
            return jsonify({"error": "No valid input provided"})

    except Exception as e:
        print("ERROR OCCURRED:\n", traceback.format_exc())
        return jsonify({"error": str(e)})

# -------------------- HELPER FUNCTIONS -------------------- #
def single_prediction(predictor, scaler, cv, text_input):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    X = cv.transform([review]).toarray()
    X_scl = scaler.transform(X)
    y_pred = predictor.predict_proba(X_scl).argmax(axis=1)[0]
    return "Positive" if y_pred == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    stemmer = PorterStemmer()
    corpus = []
    for text in data["Sentence"]:
        review = re.sub("[^a-zA-Z]", " ", str(text))
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(" ".join(review))

    X = cv.transform(corpus).toarray()
    X_scl = scaler.transform(X)
    y_preds = predictor.predict_proba(X_scl).argmax(axis=1)
    data["Predicted sentiment"] = [sentiment_mapping(y) for y in y_preds]

    # Prepare CSV in memory
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    # Generate graph as base64
    graph_base64 = get_distribution_graph_base64(data)

    return predictions_csv, graph_base64

def sentiment_mapping(y):
    return "Positive" if y == 1 else "Negative"

def get_distribution_graph_base64(data):
    fig, ax = plt.subplots(figsize=(5,5))
    counts = data["Predicted sentiment"].value_counts()
    colors = ['green', 'red']
    counts.plot.pie(autopct="%1.1f%%", colors=colors, startangle=90, shadow=True,
                    wedgeprops={"linewidth":1, "edgecolor":"black"}, ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Sentiment Distribution")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# -------------------- RUN APP -------------------- #
if __name__ == "__main__":
    app.run(port=5000, debug=True)

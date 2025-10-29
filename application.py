from flask import Flask, render_template, request
import pandas as pd
import os
import pickle
import dill
# dill is a library that allows you to serialize and deserialize Python objects
# it helps to save the model and preprocessor in a pickle file
# we don't use pickle because it can't serialize the custom transformers
# dill can serialize the custom transformers
# what does it means  is 

from src.logged.logger import logging
from src.exception import CustomException

application = Flask(__name__)
app=application


def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at: {path}")
    loader = dill if path.endswith((".pkl", ".dill")) else pickle
    with open(path, "rb") as f:
        return loader.load(f)


MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")


@app.route("/", methods=["GET"]) 
def index():
    logging.info("GET / → render form")
    return render_template("index.html")


@app.route("/predict", methods=["POST"]) 
def predict():
    try:
        logging.info("POST /predict → received form")
        # Extract form data
        form = request.form
        row = {
            "gender": form.get("gender"),
            "race_ethnicity": form.get("race_ethnicity"),
            "parental_level_of_education": form.get("parental_level_of_education"),
            "lunch": form.get("lunch"),
            "test_preparation_course": form.get("test_preparation_course"),
            "reading_score": float(form.get("reading_score")),
            "writing_score": float(form.get("writing_score")),
        }
        logging.info(f"Form row: {row}")
        df = pd.DataFrame([row])

        # Load artifacts
        preprocessor = load_artifact(PREPROCESSOR_PATH)
        model = load_artifact(MODEL_PATH)

        # Transform and predict
        X = preprocessor.transform(df)
        pred = model.predict(X)
        prediction = float(pred[0])
        logging.info(f"Prediction: {prediction}")

        return render_template("index.html", prediction=round(prediction, 2), values=row)
    except Exception as e:
        logging.info("Prediction error")
        raise CustomException(e, __import__("sys"))


if __name__ == "__main__":
    # For local dev: FLASK_RUN_HOST=0.0.0.0 FLASK_RUN_PORT=8080 flask run
    port = int(os.getenv("PORT", "8081"))
    app.run(host="0.0.0.0", port=port)



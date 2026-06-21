from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os

# ==========================
# FLASK APP
# ==========================

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================

model = tf.keras.models.load_model(
    "model/cnn_model.h5"
)

# ==========================
# UPLOAD FOLDER
# ==========================

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
# LOAD CLASSIFICATION REPORT
# ==========================

report_text = ""

report_path = "static/classification_report.txt"

if os.path.exists(report_path):

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as f:

        report_text = f.read()

else:

    report_text = """
Classification Report Not Found

Run train.py first.
"""

# ==========================
# PREDICTION FUNCTION
# ==========================

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    if prediction > 0.5:

        result = "Dog 🐶"

        confidence = round(
            prediction * 100,
            2
        )

    else:

        result = "Cat 🐱"

        confidence = round(
            (1 - prediction) * 100,
            2
        )

    return result, confidence

# ==========================
# HOME ROUTE
# ==========================

@app.route("/")
def landing():
    return render_template("home.html")


@app.route("/dashboard", methods=["GET","POST"])
def dashboard():

    result = None
    confidence = None
    img_path = None

    if request.method == "POST":

        if "image" in request.files:

            file = request.files["image"]

            if file.filename != "":

                filename = secure_filename(file.filename)

                img_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                file.save(img_path)

                result, confidence = predict_image(
                    img_path
                )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path,
        report=report_text
    )
    
# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    app.run(
        debug=True
    )
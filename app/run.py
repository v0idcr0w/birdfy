from flask import Flask, render_template, request, jsonify 
from .model import get_prediction

# imports to make image appear
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
# to start the server, run 
# flask --app app.run run --debug
# from the root dir 
@app.route('/')
def main():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error_message="No file part.")
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error_message="No file selected.")
    
    image = Image.open(file) 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    probs, prediction = get_prediction(file) 
    return render_template("index.html", prediction=prediction, probs=probs, submitted_image=base64_image)
    
    
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from utils.predictor import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload-form.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    filename = file.filename

    if not filename:
        return "No selected file"

    if allowed_file(filename):
        safe_filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        result = predict_image(filepath)

        return render_template(
            'result.html',
            image_path=filepath,
            disease=result['disease'],
            confidence=result['confidence'],
            description=result['description']
        )

    return "Invalid file format"

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

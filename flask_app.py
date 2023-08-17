from flask import Flask, request, render_template
from joblib import load
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import prewitt_h, prewitt_v
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# Load the trained model
model = load('/home/JielingChen/mysite/rf_model.joblib')

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_rgb_features(image, size=100):
    resized_img = resize(image, (size, size))
    rgb_matrix = resized_img[:, :, 0] + resized_img[:, :, 1] + resized_img[:, :, 2]
    rgb_avg_matrix = rgb_matrix / 3
    rgb_features = np.reshape(rgb_avg_matrix, (size*size))
    return rgb_features

def get_lbp_features(image, size=100):
    resized_img = resize(image, (size, size))
    image_gray = rgb2gray(resized_img)
    radius = 1
    n_points = 8 * radius
    method = 'uniform'
    lbp_matrix = local_binary_pattern(image_gray, n_points, radius, method)
    lbp_features = np.reshape(lbp_matrix, (size*size))
    return lbp_features

def get_edges_features(image, size=100):
    resized_img = resize(image, (size, size))
    image_gray = rgb2gray(resized_img)
    horizontal_edges = prewitt_h(image_gray)
    horizontal_edges = np.reshape(horizontal_edges, (size*size))
    vertical_edges = prewitt_v(image_gray)
    vertical_edges = np.reshape(vertical_edges, (size*size))
    return horizontal_edges, vertical_edges

@app.route('/')
def index():
    return render_template('Pizza or Not Pizza.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        image_pil = Image.open(file.stream)
        image = np.array(image_pil)
        rgb = get_rgb_features(image)
        lbp = get_lbp_features(image)
        h_edges, v_edges = get_edges_features(image)
        features = np.concatenate([rgb, lbp, h_edges, v_edges])
        prediction = model.predict([features])
        probabilities = model.predict_proba([features])
        probability = probabilities[0][prediction[0]] * 100
        if prediction[0] == 1:
            return 'This is a pizza! I am {:.2f}% confident!'.format(probability)
        else:
            return 'This is not a pizza! I am {:.2f}% confident!'.format(probability)
    else:
        return 'Invalid file type. Please upload an image file.'

if __name__ == '__main__':
    app.run()

import flask
import pickle
from io import BytesIO
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
import os
from werkzeug.utils import secure_filename
from flask import send_file, render_template # Make sure to import these

# --- Your existing setup ---
UPLOAD_FOLDER = os.path.join('static', 'photos')
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

device = "cuda" if is_available() else "cpu"

# --- Your Model Loading ---
resnet_model = resnet50(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(Linear(n_inputs, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 2048),
                            SELU(),
                            Dropout(p=0.4),
                            Linear(2048, 4),
                            LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)

# --- IMPORTANT ---
# Make sure you have your model file in a folder named 'models'
# e.g., Your_Project_Folder/models/bt_resnet50_model.pt
MODEL_PATH = './models/bt_resnet50_model.pt'
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at {MODEL_PATH}")
else:
    resnet_model.load_state_dict(load(MODEL_PATH, map_location=DEVICE(device)))

resnet_model.eval()

# --- Your Prediction Functions ---
def preprocess_image(image_bytes):
  transform = Compose([Resize((512, 512)), ToTensor()])
  img = Image.open(BytesIO(image_bytes))
  return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = preprocess_image(image_bytes=image_bytes)
  y_hat = resnet_model(tensor.to(device))
  class_id = argmax(y_hat.data, dim=1)
  return str(int(class_id)), LABELS[int(class_id)]

# --- Your App Routes ---
@app.route('/', methods=['GET'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('DiseaseDet.html'))

@app.route("/uimg",methods=['GET','POST'])
def uimg():
    if flask.request.method == 'GET':
        return(flask.render_template('uimg.html'))
    
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            return flask.redirect(flask.request.url)
        
        file = flask.request.files['file']

        if file.filename == '':
            return flask.redirect(flask.request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save the file to the UPLOAD_FOLDER
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read file bytes for prediction
            file.seek(0) # Go back to the start of the file
            img_bytes = file.read()
            
            class_id, class_name = get_prediction(img_bytes)
            
            # Pass the *path* to the image (not the file object) to the template
            # The template will access it at /static/photos/filename.jpg
            image_url = os.path.join('photos', filename).replace("\\", "/") # Ensure forward slashes for URL

            return(flask.render_template('pred.html', result = class_name, image_url = image_url))
        
        else:
            return flask.redirect(flask.request.url)

# --- NEW ROUTE: Download the Model ---
@app.route('/download/model')
def download_model():
    """
    Provides the model file for download.
    """
    try:
        return send_file(MODEL_PATH, as_attachment=True, download_name='bt_resnet50_model.pt')
    except Exception as e:
        print(f"Error sending file: {e}")
        return render_template('error.html', error="Model file not found on server."), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error=str(error)), 500

if __name__ == '__main__':
   	app.run(debug=True)

import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import shutil

app = Flask(__name__)

# Dictionary for 8 classes
dic = {
    0: 'Class1', 
    1: 'Class2', 
    2: 'Class3', 
    3: 'Class4', 
    4: 'Class5', 
    5: 'Class6', 
    6: 'Class7', 
    7: 'Class8'
}

# Load your pre-trained 8-class model
model = load_model('model.h5')

def predict_label(img_path):
    # Load and preprocess the image
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    
    # Make a prediction using the model
    p = model.predict(i)
    
    # Get the class with the highest probability
    predicted_class = dic[np.argmax(p)]
    
    # Get the highest confidence score (in percentage)
    confidence = np.max(p) * 100  
    
    return predicted_class, confidence

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        # Get the uploaded image
        img = request.files['my_image']
        
        # Temporary directory for uploads
        temp_dir = 'temp/'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        img_path_temp = os.path.join(temp_dir, img.filename)
        img.save(img_path_temp)
        
        # Move the file to the static directory
        img_path_static = os.path.join('static', img.filename)
        shutil.move(img_path_temp, img_path_static)

        # Get prediction and confidence level
        p, confidence = predict_label(img_path_static)

        # Render the result on the page
        return render_template("index.html", prediction=p, confidence=confidence, img_path=img.filename)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

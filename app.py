import numpy as np
import pickle
import json
from werkzeug import secure_filename
from flask import Flask,flash, request, jsonify, render_template, url_for
from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.models import load_model
from PIL import Image
import cv2
from dbcursor import conn
from blobconn import blob
app = Flask(__name__)

global model,base,tokenizer,cursor,blobservice,err,cnxn
model = load_model('resources/modelmain.h5')
tokenizer = pickle.load(open('resources/tokenizer.pkl','rb'))
base = load_model('resources/modelIncp.h5')
err = True
try:
    cnxn = conn()
    cursor = cnxn.cursor()
    blobservice = blob()
except:
    err = False

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(filename,base):
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    feature = base.predict(image)
    print(feature.shape)
    return feature

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence])
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contrib',methods=['GET','POST'])
def contrib():
    if request.method=='GET':
        return render_template('contrib.html')
    else:
        email = request.form['email']
        caption = request.form['caption']
        imagesub = request.files['image']
        imagesub.save(secure_filename("temp_img.jpg"))
        container_name = 'images'
        if err:
            try : 
                cursor.execute("SELECT max(id) FROM details;") 
                row = cursor.fetchone()
                if row[0]==None:
                    nametemp = 1
                else:
                    nametemp = row[0]+1
                query = "INSERT INTO details (id,caption,email) VALUES ("+str(nametemp)+",'"+caption+"','"+email+"');"
                print(query)
                cursor.execute(query);
                nametemp = str(nametemp)+".jpg"
                img = "./temp_img.jpg"
                blobservice.create_blob_from_path(container_name, nametemp, img)
                cnxn.commit()
                return render_template('contrib.html', message = "Data added successfully!", type="bg-success")
            except:
                return render_template('contrib.html', message = "Unable to add Data currently!", type="bg-danger")
        else:
            return render_template('contrib.html', message = "Unable to add Data currently!", type="bg-danger")

@app.route('/predict_api',methods=['POST','GET'])
def predict_api():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("temp.jpg",img)
    photo = extract_features('temp.jpg',base)
    prediction = generate_desc(model,tokenizer,photo,74)
    print(prediction)
    return json.dumps({"pred":prediction})

if __name__ == '__main__':
    print("Flask app running....")
    print("Initialised, waiting for connection.")
    app.run(host='0.0.0.0',debug=True)

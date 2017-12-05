from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
import pickle as pkl
import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
import requests
from google.cloud import vision
import json
import genJson
import h5py
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
    global model
    x = load_img(file)
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

import json
import math

def angle(jsonFiy):
    data = jsonFiy
    
    left_eye = data["responses"][0]["faceAnnotations"][0]["landmarks"][0]
    right_eye = data["responses"][0]["faceAnnotations"][0]["landmarks"][1]
    mouth_left = data["responses"][0]["faceAnnotations"][0]["landmarks"][10]
    mouth_right = data["responses"][0]["faceAnnotations"][0]["landmarks"][11]
    
    face_feature = [left_eye, right_eye, mouth_left, mouth_right]
    
    left_eye_cor_y = face_feature[0]['position']['y']
    right_eye_cor_y = face_feature[1]['position']['y']
    left_eye_cor_x = face_feature[0]['position']['x']
    right_eye_cor_x = face_feature[1]['position']['x']
    
    mouth_left_cor_y = face_feature[2]['position']['y']
    mouth_right_cor_y = face_feature[3]['position']['y']
    mouth_left_cor_x = face_feature[2]['position']['x']
    mouth_right_cor_x = face_feature[3]['position']['x']
    
    slope_eyes = ((left_eye_cor_y - right_eye_cor_y)/(left_eye_cor_x - right_eye_cor_x))
    slope_mouth = ((mouth_left_cor_y - mouth_right_cor_y)/(mouth_left_cor_x - mouth_right_cor_x))
    
    if slope_eyes > 0:
        alpha = 180 - math.atan(1/slope_eyes)
    elif slope_eyes < 0:
        alpha = math.atan(1/slope_eyes)
    
    if slope_mouth > 0:
        beta = 180 - math.atan(1/slope_mouth)
    elif slope_mouth < 0:
        beta = math.atan(1/slope_mouth)

    ang = abs(alpha - beta)

    return(ang)


app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == "GET":
        im = request.args.get('photo')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        imagedata = base64.b64decode(im)
        filename = 'image.jpg'
        with open(filename,'wb')as f:
            f.write(imagedata)
        prediction = predict(filename)
        print(prediction)
        os.system("sudo python genJson.py -i /home/frankwang971120/trying.txt -o /home/frankwang971120/vision.json")
        data2 = open('/home/frankwang971120/vision.json').read()
        print("---------------------------RESPONSE---------------------------")
        response = requests.post(url='https://vision.googleapis.com/v1/images:annotate?key=AIzaSyBmObSUnQJbM1WdWb0-IfCYY8PBH6JdVC8',data=data2,headers={'Content-Type': 'application/json'})
        
        angle_drop = angle(response.json())
        if angle_drop > 90:
            angle_drop = 180 - angle_drop
            
        print(response.text)
        print('-----------------------------------')
        angle_drop = angle_drop * 1000.0000
        stroke_risk = False
        if angle_drop > 25:
            stroke_risk = True
        print(angle_drop)
        print('--HHHHHHHHHH--')
        #dic = dict(response.text)
        #print(dic)

        print("---------------------------RESPONSE ENDS---------------------------")
        res1 , conf1, res2, conf2 = "Unhealthy",0.99,"Healthy",0.01
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        return jsonify({"result":prediction,"stroke_risk":stroke_risk}),200
    return jsonify({"result":prediction,"stroke_risk":stroke_risk}),200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = False)


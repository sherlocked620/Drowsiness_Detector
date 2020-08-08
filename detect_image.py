#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:58:47 2020

@author: Kriti
"""

import joblib
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import face_recognition
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils import paths
import os

print("[INFO] loading feature extractor model...")
feature_extractor = load_model('model_feature')
feature_extractor.compile(optimizer = 'Adam',loss = 'binary_crossentropy',metrics = ['acc'])
    
print('[INFO] loading xgbooster model.....')
xg_model = joblib.load('drowsy_xg-model.sav')

def DetectImage(imgpath,feature_extractor,xg_model):
    frame = cv2.imread(imgpath)
    img = frame.copy()
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face = img[top:bottom, left:right]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)/255.
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        feature_vector = feature_extractor.predict(face)
        pred = xg_model.predict(feature_vector)
        label = "Yawn" if pred<0.5  else "No Yawn"
        color = (0, 255, 0) if label == "No Yawn" else (0, 0, 255)
        # include the probability in the label
        cv2.putText(frame, label, (left + 6, bottom - 6),cv2.FONT_HERSHEY_SIMPLEX, 1.0,color, 1)
        cv2.rectangle(frame, (left, top), (right, bottom),color, 2)
        print(label+ str(pred))

    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
         #if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    return frame


print("[INFO] loading Processed Image...")
test_path = 'sample/22.jpg'
DetectImage(test_path,feature_extractor,xg_model)



# -*- coding: utf-8 -*-

import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from PIL import Image

# This key will serve all examples in this document.
KEY = "<secret>"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "<secret>"

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Group image for testing against
from_dir = "../dataset/data/吉本坂/"
to_dir = "./yoshimotozaka/"

files = glob.glob(from_dir+"*.jpg")

emotion_dict = {}
emotion_dict["anger"]=0
emotion_dict["contempt"]=0
emotion_dict["disgust"]=0
emotion_dict["happiness"]=0
emotion_dict["neutral"]=0
emotion_dict["sadness"]=0
emotion_dict["surprise"]=0

for file in files:
    image = open(file, 'r+b')
    print(file)

    time.sleep(5)
    # Detect faces
    face_ids = []
    # We use detection model 2 because we are not retrieving attributes.
    faces = face_client.face.detect_with_stream(image, detectionModel='detection_02',return_face_attributes=["emotion"])

    if not faces:
	    continue

    for face in faces:
        face_ids.append(face.face_id)
        width = face.face_rectangle.width
        height = face.face_rectangle.height
        left = face.face_rectangle.left
        top = face.face_rectangle.top

        # triming
        im = Image.open(file)
        im_crop = im.crop((left, top, left+width, top+height))
        im_crop.save(to_dir + face.face_id+'.jpg', quality=95)

        emotion = face.face_attributes.emotion
        emotion_dict["anger"]+=emotion.anger
        emotion_dict["contempt"]+=emotion.contempt
        emotion_dict["disgust"]+=emotion.disgust
        emotion_dict["happiness"]+=emotion.happiness
        emotion_dict["neutral"]+=emotion.neutral
        emotion_dict["sadness"]+=emotion.sadness
        emotion_dict["surprise"]+=emotion.surprise
    
    max_emotion = max(emotion_dict, key=emotion_dict.get)
    
    #print("{}のトリミングが完了しました".format(test))
    print("{}人".format(len(faces)))
    print("感情判定: {}".format(max_emotion))

# -*- coding: utf-8 -*-

from flask import Flask, Response, request, render_template, send_file, jsonify
import json
import base64
from base64 import b64encode
from base64 import b64decode
from io import BytesIO

import io
import glob
import os
import uuid
import time
import uuid
import requests
import random
import urllib
import cv2
from urllib.parse import urlparse
from datauri import DataURI
import numpy as np

from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials

# FACE API
KEY = "<secret>"
FACE_ENDPOINT = "<secret>"

# CUSTOM VISION
CUSTOM_ENDPOINT = "<secret>"
TRAINING_KEY = "<secret>"
PREDICTION_KEY = "<secret>"
PREDICTION_RESOURCE_ID = "<secret>"
PROJECT_ID = "<secret>"
ITERATION = "<secret>"

app = Flask(__name__)

def get_faces(image):

    face_client = FaceClient(FACE_ENDPOINT, CognitiveServicesCredentials(KEY))
    

    #time.sleep(2)
    
    # We use detection model 2 because we are not retrieving attributes.
    face_ids = []
    faces = face_client.face.detect_with_stream(image, detectionModel="detection_02",return_face_attributes=["emotion"])
    print("{}人の顔が検知されました".format(len(faces)))

    return faces

def trim_face(file):

    image = open(file, "r+b")
    faces = get_faces(image)

    if not faces:
        print("{}の顔が認識されませんでした".format(file))
        return [], None, []

    #if len(faces) > 1:
    #    print("認識される顔は1つまでです")
    #    exit()

    pic_faces = []
    emotion_dict = {}
    emotion_dict["anger"]=0
    emotion_dict["contempt"]=0
    emotion_dict["disgust"]=0
    emotion_dict["happiness"]=0
    emotion_dict["neutral"]=0
    emotion_dict["sadness"]=0
    emotion_dict["surprise"]=0
    
    for face in faces:

        width = face.face_rectangle.width
        height = face.face_rectangle.height
        left = face.face_rectangle.left
        top = face.face_rectangle.top

        # triming
        im = Image.open(image)
        im_crop = im.crop((left, top, left+width, top+height))

        dir_path = "./pic_face/"
        uid = str(uuid.uuid4())
        filename = dir_path + uid + ".jpg"
        im_crop.save(filename, quality=95)
        pic_faces.append(filename)

        emotion = face.face_attributes.emotion
        emotion_dict["anger"]+=emotion.anger
        emotion_dict["contempt"]+=emotion.contempt
        emotion_dict["disgust"]+=emotion.disgust
        emotion_dict["happiness"]+=emotion.happiness
        emotion_dict["neutral"]+=emotion.neutral
        emotion_dict["sadness"]+=emotion.sadness
        emotion_dict["surprise"]+=emotion.surprise
    
    max_emotion = max(emotion_dict, key=emotion_dict.get)
    
    print("{}のトリミングが完了しました".format(file))
    print("感情判定: {}".format(max_emotion))

    return pic_faces, max_emotion, faces


def judge_sakamichi(faces):

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(CUSTOM_ENDPOINT, prediction_credentials)
    publish_iteration_name = ITERATION
    project_id = PROJECT_ID

    #time.sleep(2)
    
    judgements = {}
    for face in faces:
        print(" ")
        with open(face, "rb") as image_contents:
            results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())

        # Display the results.
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                ": {0:.2f}%".format(prediction.probability * 100))
        
            if prediction.tag_name not in judgements:
                judgements[prediction.tag_name] = 0
            judgements[prediction.tag_name] += prediction.probability * 100

    judgement = max(judgements, key=judgements.get)

    return judgement

def resize_image(j_face, o_face, pic_face):

    # 縦横取得
    j_height = j_face.face_rectangle.height
    j_width = j_face.face_rectangle.width
    o_height = o_face.face_rectangle.height
    o_width = o_face.face_rectangle.width

    # 比率
    height_ratio = float(j_height)/o_height
    width_ration = float(j_width)/o_width

    # リサイズ
    image = cv2.imread(pic_face, cv2.IMREAD_UNCHANGED) 
    resized = cv2.resize(image,(int(o_height*height_ratio),int(o_width*width_ration)))

    return resized   


def overlay_image(colla_image, resized_image, j_face): 
    #print()
    re_height, re_width = resized_image.shape[:2]
    #print(re_height, re_width)

    # BGRAからRGBAへ変換
    colla_image_RGBA = cv2.cvtColor(colla_image, cv2.COLOR_BGR2RGB)
    resized_image_RGBA = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2RGBA)

    # PILに変換
    colla_image_PIL = Image.fromarray(colla_image_RGBA)
    resized_image_PIL = Image.fromarray(resized_image_RGBA)

    # RGBAモードに変更
    colla_image_PIL = colla_image_PIL.convert("RGBA")
    resized_image_PIL = resized_image_PIL.convert("RGBA")

    # 合成
    left = j_face.face_rectangle.left
    top = j_face.face_rectangle.top
    #print(left, top)
    tmp = Image.new("RGBA", colla_image_PIL.size, (255, 255, 255, 0))
    tmp.paste(resized_image_PIL, (left, top), resized_image_PIL)
    result = Image.alpha_composite(colla_image_PIL, tmp)

    return  cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)


def create_collapic(pic_faces, judgement, o_faces, emotion):
    
    #judgement = "nogi"
    from_dir = "./pic_jacket/" + judgement + "/"
    jackets = glob.glob(from_dir+"*.jpg")

    #jacketの選択(random)
    jacket = random.choice(jackets)
    #jacket = from_dir + "25-1.jpg"

    #song idの導出
    song_id = jacket
    song_id = song_id[len(from_dir):-4]

    #jacket画像の顔検知
    j_image = open(jacket, "r+b")
    j_faces = get_faces(j_image)
    
    # resizeをしてcollage
    colla_image = cv2.imread(jacket)  # collage対象
    for i in range(len(j_faces)):

        if len(j_faces) > 2:
            if np.random.rand() < 0.25: #25%の確率で雑コラをスキップ
                print("雑コラがスキップされました!")
                continue

        # resize
        j_face = j_faces[i]
        index = random.choice(range(len(o_faces)))  # random選択
        o_face = o_faces[index]
        pic_face = pic_faces[index]
        resized_image = resize_image(j_face, o_face, pic_face)
        #cv2.imwrite("resize.png", resized_image)

        # collage
        colla_image = overlay_image(colla_image, resized_image, j_face)

    height = colla_image.shape[1]
    #param = random.choice([(255,255,255), (0,0,0)])
    param = (random.randrange(256), random.randrange(256), random.randrange(256))
    #cv2.putText(colla_image, "You look " + emotion + ".", (10, 55), cv2.FONT_HERSHEY_PLAIN, int(height/300), param, 5, cv2.LINE_AA)
    cv2.putText(colla_image, emotion[0].upper() + emotion[1:], (10, 55), cv2.FONT_HERSHEY_PLAIN, int(height/300), param, 5, cv2.LINE_AA)
    dir_path = "./pic_colla/"
    uid = str(uuid.uuid4())
    filename = dir_path + uid + ".jpg"
    cv2.imwrite(filename, colla_image)
    print("colla画像 {} を生成しました".format(filename))
    print("song_id: {}".format(song_id))

    return filename, song_id


@app.route("/collage", methods=["POST"])
def collage():
   
    dir_path = "./pic_all/"

    # toypo
    json_data = request.get_json() # POSTされたjsonを取得
    img = json_data["img"]

    # python
    #json_data = request.get_json() # POSTされたjsonを取得
    #dict_data = json.loads(json_data) # jsonを辞書に変換
    #img = dict_data["img"] # base64を取り出す # str

    img_b64 = b64decode(img.encode()) # base64に変換された画像データを元のバイナリデータに変換 # bytes

    # uidの生成
    uid = str(uuid.uuid4())
    o_file = dir_path + uid + ".jpg"
    with open(o_file, "wb") as f:
        f.write(img_b64)

    pic_faces, emotion, faces = trim_face(o_file) # 顔画像のパスを取得
    if not faces:   #顔が検知されなかった場合
        return jsonify({'message': 'Cannot detect one or more faces'}), 400
    #print(pic_faces)
    #print(emotion)

    judgement = judge_sakamichi(pic_faces)
    print("坂道判定: {}".format(judgement))

    c_file, song_id = create_collapic(pic_faces, judgement, faces, emotion)

    # data uriの作成
    with open(c_file, "rb") as f:
        data = f.read()
    img_b64=b64encode(data)
    img_str = img_b64.decode("utf-8")
    data_uri = "data:image/jpeg;base64,{}".format(img_str)
    uri = DataURI(data_uri)
    
    json_data ={
        "img": uri,
        "team_type": judgement,
        "song": song_id,
        "emotion": emotion
    }

    #return send_file(io.BytesIO(uri.data), mimetype=uri.mimetype)
    return json.dumps(json_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
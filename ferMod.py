from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
import ffmpeg    

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
    #return round(int(rotate) / 90.0) * 90
    #if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        #rotateCode = cv2.ROTATE_90_CLOCKWISE
    #elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        #rotateCode = cv2.ROTATE_180
    #elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        #rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    #return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Frustrated", "Disagree", "Tense", "Happy", "Frustrated", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(
                loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("fps")
args = parser.parse_args()

if args.source != 'webcam':
    cap = cv2.VideoCapture(os.path.abspath(args.source))
    #rotateCode = check_rotation(os.path.abspath(args.source))
elif args.source == 'webcam':
    cap = cv2.VideoCapture(0)


faceCascade = cv2.CascadeClassifier('models//haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))


def getdata():
    grabbed, fr = cap.read()
    #if rotateCode is not None:
        #print(rotateCode)
        #fr = correct_rotation(fr, rotateCode)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.25, 4)
    return faces, fr, gray


def start_app(cnn):
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 255, 0), 1)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Facial Emotion Recognition', fr)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)

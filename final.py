from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
import ffmpeg
from openpyxl import  Workbook
from openpyxl.drawing.image import Image
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
    # return round(int(rotate) / 90.0) * 90
    # if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
    # rotateCode = cv2.ROTATE_90_CLOCKWISE
    # elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
    # rotateCode = cv2.ROTATE_180
    # elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
    # rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    # return rotateCode


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
    # rotateCode = check_rotation(os.path.abspath(args.source))
elif args.source == 'webcam':
    cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('models//haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))


def getdata():
    grabbed, fr = cap.read()
    # if rotateCode is not None:
    # print(rotateCode)
    # fr = correct_rotation(fr, rotateCode)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.25, 4)
    return faces, fr, gray


def overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if (x2 >= w1 or x1 >= w2):
        return False
    if (y2 >= h1 or y1 >= h2):
        return False

    return True


def areaOfIntersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    aoi = (min(w1, w2) - max(x1, x2)) * (min(h1, h2) - max(y1, y2))
    rect1Area = (w1 - x1) * (h1 - y1)
    prcntAOI = (aoi / rect1Area) * 100
    return prcntAOI


def liesInside(rect, pt):
    p, q = pt
    x, y, w, h = rect
    if (p <= w and p >= x and q <= h and p >= y):
        return True
    else:
        return False


def start_app(cnn):
    boxes = {}
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z',
               'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM'
               ]

    clm = 1
    frcnt = 0
    face_cnt = 0
    erf = 4 #excel row where frame count starts.
    book = Workbook()
    sheet = book.active
    sheet['A1'] = "frameNo"

    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        cv2.putText(fr, "Frame " + str(frcnt), (20, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        sheet['A'+str(frcnt+4)] = frcnt

        if len(boxes) == 0:
            for (x, y, w, h) in faces:
                crop_face = fr[y:y + h, x:x + w]
                x_o, y_o, w_o, h_o = x - 0, y - 0, x + w + 0, y + h + 0
                name = 'face'+str(face_cnt)
                boxes[(x_o, y_o, w_o, h_o)] = {"Frustrated": 0, "Disagree": 0,
                                               "Tense": 0, "Happy": 0,
                                               "Surprise": 0, "Neutral": 0,
                                               "show_img":crop_face,"face_name":name}
                sheet[columns[clm]+ str(1)] = str((x_o, y_o, w_o, h_o))
                sheet[columns[clm] + str(2)] = name
                cv2.imwrite(name+'.png',crop_face)
                img_in_sheet = Image(name+'.png')
                sheet.add_image(img_in_sheet,columns[clm] + str(3) )
                clm+=1
                face_cnt += 1
        else:
            # pass
            for (x, y, w, h) in faces:
                isNew = True
                crop_face = fr[y:y + h, x:x + w]
                x_o, y_o, w_o, h_o = x - 0, y - 0, x + w + 0, y + h + 0
                box_length = len(boxes)
                for (p, q, r, s) in boxes:
                    ifo = p_new, q_new, r_new, s_new = p - 25, q - 25, p + 25, q + 25

                    if (overlap((x_o, y_o, w_o, h_o),(p,q,r,s))):
                        if(areaOfIntersection((x_o, y_o, w_o, h_o),(p,q,r,s)) > 20):
                            isNew = False

                # print(compared_faces)
                if (isNew):
                    name = 'face' + str(face_cnt)
                    boxes[(x_o, y_o, w_o, h_o)] = {"Frustrated": 0, "Disagree": 0,
                                                   "Tense": 0, "Happy": 0,
                                                   "Surprise": 0, "Neutral": 0,
                                                   "show_img":crop_face,"face_name":name}
                    sheet[columns[clm]+str(1)] = str((x_o, y_o, w_o, h_o))
                    sheet[columns[clm] + str(2)] = name
                    cv2.imwrite(name + '.png', crop_face)
                    img_in_sheet = Image(name + '.png')
                    sheet.add_image(img_in_sheet, columns[clm] + str(3))
                    clm += 1
                    face_cnt+=1


        for each in boxes:
            cv2.rectangle(fr, (each[0], each[1]), (each[2], each[3]), (0, 255, 0), 1)
            cv2.putText(fr, boxes[each]['face_name'], (each[2]-40, each[3]+20 ),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            # cv2.rectangle(fr, (each[0] - 25, each[1] - 25), (each[0] + 25, each[1] + 25), (0, 0, 255), 1)

        for (x, y, w, h) in faces:
            face = (x,y,w,h)
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            # print(pred)
            # print(frcnt)
            # print(face)
            face_new = face[0], face[1], face[0] + face[2], face[1] + face[3]
            for a_face in boxes:
                if overlap(face_new,a_face):
                    for ln in range(len(boxes)):
                        if(sheet[columns[ln+1]+str(2)].value == boxes[a_face]["face_name"]):
                            sheet[columns[ln+1]+str(erf)] = pred
                    # if pred == "Happy":
                    #     boxes[a_face]["Happy"] += 1
                    # if pred == "Frustrated":
                    #     boxes[a_face]["Frustrated"] += 1
                    # if pred == "Disagree":
                    #     boxes[a_face]["Disagree"] += 1
                    # if pred == "Tense":
                    #     boxes[a_face]["Tense"] += 1
                    # if pred == "Surprise":
                    #     boxes[a_face]["Surprise"] += 1
                    # if pred == "Neutral":
                    #     boxes[a_face]["Neutral"] += 1


            cv2.putText(fr, pred, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 255, 0), 1)

        if cv2.waitKey(1) == 27:

            break
        # cv2.imwrite('newdir/' + 'frame' + str(frcnt) + '.jpg', fr)

        cv2.imshow('Facial Emotion Recognition', fr)
        frcnt += 1
        erf +=1

        with open('res.txt','w') as res_file:
            for key in boxes.keys():
                res_file.write("%s,%s\n" % (key, boxes[key]))


        # if frcnt == 25:
        #     with open('facesRecorded.txt', 'w') as text_file:
        #         text_file.write("Faces detected in frame"+str(frcnt)+ ":\n")
        #         for (a1,b1,c1,d1) in faces:
        #             each_face = (a1,b1,c1+a1,d1+b1)
        #             text_file.write(str(each_face)+",\n")
        #         text_file.write("###############################\n")
        #         text_file.write("Originally Saved Faces:" +"\n")
        #         for rec_face in boxes:
        #             text_file.write(str(rec_face)+",\n")

        # name = 0
        # for every_face in boxes:
        #     filename = 'face'+str(name)+'.png'
        #     face_img = boxes[every_face]['show_img']
        #     cv2.imwrite(filename,face_img)
        #     name+=1

    book.save("recorded_face_data.xlsx")
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)

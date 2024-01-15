import cv2
import goi_ham as gh
import paho.mqtt.client as paho
import numpy as np
import face_recognition
import os
import gtts
from playsound import playsound
from gtts import gTTS 
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 100)  # Set the speech rate


flag = 0
sub_topic = "QR_check"

#path = r'D:\document\95IDEAL\code\QR_checkin\QR_Event-main\facepath'
path = r"\\192.168.1.33\facepath"
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)
count = len(mylist)


for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def findEndcodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeList = findEndcodings(images)
print("Encodding Complete")
print(len(encodeList))
cap = cv2.VideoCapture(0)
before = os.listdir(path)
len1 = len(before)


def on_publish(client, userdata, mid):
    print("mid: "+str(mid))
 

    
def on_message(client, userdata, msg):
    if(msg.payload.decode("utf-8") == "0"):
        if flag == 1:
            engine.say("Xin vui lòng quét mã vạch")
            engine.runAndWait()
            


client = paho.Client()

client.on_message = on_message


client.connect('broker.mqttdashboard.com', 1883)
client.subscribe(sub_topic, qos=1)



while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    after = os.listdir(path)
    len2 = len(after)
    if len2 > len1:
        # Find the new image by comparing the lists
        new_image = set(after) - set(before)
        # Get the full path of the new image
        if new_image:
            new_image_path = os.path.join(path, new_image.pop())
            img_new = cv2.imread(new_image_path)
            img = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
            file_name, _ = os.path.splitext(os.path.basename(new_image_path))
            classnames.append(file_name)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print(classnames)
            print(len(encodeList))
        else:
            print("No new image added to the folder.")
        len1 = len2
    faceLocFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS, faceLocFrame)

    for encodeFace, faceLoc in zip(encodeFrame, faceLocFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDis = face_recognition.face_distance(encodeList, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            # print(name)
            mydb = gh.connect_database()
            mycursor = mydb.cursor()
            mycursor.execute("SELECT Ten FROM yourtablename WHERE QRID = %s", (name,))
            available = mycursor.fetchall()
            var = list(available[0])
            print(var[0])
            flag = 0
        else:
            flag = 1
            print("Khong co trong danh sach")
    cv2.imshow("frame", img)
    # Check for new messages every 0.1 seconds
    client.loop(timeout=0.1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
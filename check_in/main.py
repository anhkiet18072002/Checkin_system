import errno
import multiprocessing
from multiprocessing import Process, Manager
import cv2
import dlib
import serial
import function_speak as fs
import numpy as np
import face_recognition
import os

# from playsound import playsound
from google_speech import Speech
from gpiozero import Button
from signal import pause
import time
from queue import Queue


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_gaze_ratio(eye_points, facial_landmarks, gray):
    mask = np.zeros_like(gray)
    eye_region = np.array(
        [
            (facial_landmarks.part(point).x, facial_landmarks.part(point).y)
            for point in eye_points
        ],
        np.int32,
    )
    cv2.polylines(mask, [eye_region], isClosed=True, color=255, thickness=2)
    cv2.fillPoly(mask, [eye_region], color=255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]

    # Verify the ROI is valid
    if gray_eye.size == 0:
        return None

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # Check if thresholding was successful
    if threshold_eye is None:
        return None

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0 : int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2) : width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def face_match(encodeList, encodeFace, tolerance=0.4):
    match = False
    matchIndex = None
    for i, known_face_encoding in enumerate(encodeList):
        distance = euclidean_distance(known_face_encoding, encodeFace)
        if distance <= tolerance:
            match = True
            matchIndex = i
            break
    return match, matchIndex


def process1(queue, check, string_var):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        framecpy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray)
            right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)
            if left_eye_ratio is None or right_eye_ratio is None:
                continue

            gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if gaze_ratio <= 1.2:
                gaze_direction = "Looking at camera"
                # Save the face image
                if check.value == 1:
                    save_path = (
                        "/home/pi/QR/main/facepath/" + str(string_var[0]) + ".jpg"
                    )
                    if not os.path.exists(os.path.dirname(save_path)):
                        try:
                            os.makedirs(os.path.dirname(save_path))
                        except OSError as exc:  # Guard against race condition
                            if exc.errno != errno.EEXIST:
                                raise
                    success = cv2.imwrite(save_path, framecpy)
                    if success:
                        print(f"Image saved successfully at {save_path}")
                        fs.update_data(str(string_var[0]), save_path)
                        check.value = 0
                    else:
                        print(f"Failed to save image at {save_path}")
            else:
                gaze_direction = "Not looking at camera"

            cv2.putText(
                frame,
                gaze_direction,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        queue.put(frame)
        cv2.imshow("Processed Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def process2(check, string_var, voice):
    ser = serial.Serial(
        "/dev/ttyACM0", 9600, timeout=2
    )  # Thiáº¿t láº­p timeout á»Ÿ Ä‘Ă¢y
    try:
        while True:
            barcode_data = ser.readline().decode().strip()
            if barcode_data:
                print(f"Ma vach doc duoc la: {barcode_data}")
                check_var = fs.check_available(barcode_data)
                if check_var == None:
                    text = "QR not valid. Please try again."
                    lang = "en"
                    notval = Speech(text, lang)
                    notval.play()
                else:
                    print("QR ton tai trong he thong.")
                    check.value = 1
                    voice.value = 1
                    string_var[:] = [barcode_data]

    finally:
        ser.close()  # Close the serial connection when done


def process3(string_var, voice):
    while True:
        if voice.value == 1:
            fs.check(str(string_var[0]))
            voice.value = 0


def display_frames(queue):
    while True:
        if not queue.empty():
            frame = queue.get()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def process4(queue):
    path = r"/home/pi/QR/main/facepath"
    images = []
    classnames = []
    mylist = os.listdir(path)
    print(mylist)
    for cl in mylist:
        curImg = cv2.imread(f"{path}/{cl}")
        images.append(curImg)
        classnames.append(os.path.splitext(cl)[0])
    print(classnames)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodes = face_recognition.face_encodings(img)
            if encodes:  # Ensure the list is not empty
                encodeList.append(encodes[0])
        return encodeList

    encodefile_path = "Encodefile.npy"
    if not os.path.exists(encodefile_path) or os.path.getsize(encodefile_path) == 0:
        print("File does not exist")
        encodeList = findEncodings(images)
        np.save(encodefile_path, encodeList)
        print("Encoding complete")
        print(len(encodeList))
    else:
        print("File exists")
        encodeList = list(np.load(encodefile_path, allow_pickle=True))

    before = os.listdir(path)
    len1 = len(before)

    button = Button(2)  # Initialize button on GPIO pin 2
    debounce_time = 0.1  # Debounce time in seconds
    last_button_press = time.time()
    flag = 0
    while True:
        if not queue.empty():
            img = queue.get()
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
                    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
                    file_name, _ = os.path.splitext(os.path.basename(new_image_path))
                    classnames.append(file_name)
                    encode = face_recognition.face_encodings(img_new)[0]
                    encodeList.append(encode)
                    np.save(encodefile_path, encodeList)
                    print(classnames)
                    print(len(encodeList))
                else:
                    print("No new image added to the folder.")
                len1 = len2
            if (
                button.is_pressed
                and (time.time() - last_button_press > debounce_time)
                and (flag == 0)
            ):
                last_button_press = time.time()  # Update the last button press time
                flag = 1
                faceLocFrame = face_recognition.face_locations(imgS)
                encodeFrame = face_recognition.face_encodings(imgS, faceLocFrame)
                encodeList_save = np.load(encodefile_path, allow_pickle=True)

                for encodeFace, faceLoc in zip(encodeFrame, faceLocFrame):
                    match, matchIndex = face_match(encodeList_save, encodeFace)
                    if match:
                        name = classnames[matchIndex].upper()
                        mydb = fs.connect_database()
                        mycursor = mydb.cursor()
                        mycursor.execute(
                            "SELECT Fullname FROM yourtablename WHERE QRID = %s",
                            (name,),
                        )
                        available = mycursor.fetchall()
                        var = list(available[0])
                        print(var[0])
                        text = "Welcome " + var[0] + ". Please come in."
                        lang = "en"
                        comein = Speech(text, lang)
                        comein.play()
                    else:
                        print("Khong co trong danh sach. Vui long quet ma QR")
                        text = "Your face is not registered. Please scan your QR code."
                        lang = "en"
                        notval2 = Speech(text, lang)
                        notval2.play()
                flag = 0
                # Check for new messages every 0.1 seconds
            queue.put(img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break


if __name__ == "__main__":
    with Manager() as manager:

        count = multiprocessing.Value("i", 0)
        check = multiprocessing.Value("i", 0)
        voice = multiprocessing.Value("i", 0)
        string_var = manager.list([""])
        queue = multiprocessing.Queue()

        p1 = multiprocessing.Process(target=process1, args=(queue, check, string_var))
        p2 = multiprocessing.Process(target=process2, args=(check, string_var, voice))
        p3 = multiprocessing.Process(target=process3, args=(string_var, voice))
        p4 = multiprocessing.Process(target=process4, args=(queue,))
        display_process = Process(target=display_frames, args=(queue,))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        display_process.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        display_process.join()

        print("Tiến trình kết thúc")

import multiprocessing 
from multiprocessing import Process, Manager
import cv2
import dlib
from imutils import face_utils
import serial
import time
import goi_ham as gh

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def process1(queue, count, check, string_var, voice):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        framecpy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if count.value >= 20:
                # Save the face image
                save_path = "facepath\\" + str(string_var[0]) + ".jpg"
                cv2.imwrite(save_path, framecpy)

                # Update data and reset counters 
                gh.update_data(str(string_var[0]), save_path)
                count.value = 0
                check.value = 0

        queue.put(frame)
        cv2.imshow("Processed Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

def process2(queue, count, check, string_var,voice):
    ser = serial.Serial('COM6', 9600, timeout=2)  # Thiết lập timeout ở đây
    try:
        while True:
            barcode_data = ser.readline().decode().strip()
            if barcode_data:
                print(f'Mã vạch đã đọc được: {barcode_data}')
                check_var = gh.check_available(barcode_data)
                if check_var == None:
                    print("QR khong ton tai trong he thong.")
                else:
                    print("QR ton tai trong he thong.") 
                    check.value = 1
                    voice.value = 1
                    string_var[:] = [barcode_data]
                    
            while check.value == 1:
                if not queue.empty():
                    frame = queue.get()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                        minNeighbors=5, minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE)

                    if len(rects) == 0:
                        count.value = 0
                    else:
                        for (x, y, w, h) in rects:
                            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            
                            if (len(shape[36:48]) >= 6) and (len(shape[48:68]) >= 20):
                                
                                count.value += 1
                            else:
                                
                                count.value = 0
                    print(count.value)

                    # Send the processed frame to the main process for display
                    queue.put(frame)
        
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        ser.close()  # Close the serial connection when done
def process3(string_var, voice):
    while True:
        if voice.value == 1:
            gh.check(str(string_var[0]))
            voice.value = 0
    
def display_frames(queue):
    while True:
        if not queue.empty():
            frame = queue.get()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
if __name__ == "__main__":
    with Manager() as manager:        
        count = multiprocessing.Value('i', 0)
        check = multiprocessing.Value('i', 0)
        voice = multiprocessing.Value('i', 0)
        string_var = manager.list([""])
        queue = multiprocessing.Queue()

        p1 = multiprocessing.Process(target=process1, args=(queue, count, check, string_var,voice))
        p2 = multiprocessing.Process(target=process2, args=(queue, count, check, string_var,voice))
        p3 = multiprocessing.Process(target=process3, args=(string_var,voice))
        display_process = Process(target=display_frames, args=(queue,))

        p1.start()
        p2.start()
        p3.start()
        display_process.start()

        p1.join()
        p2.join()
        p3.join()
        display_process.join()

        print("Tiến trình đã kết thúc")

    cv2.destroyAllWindows()
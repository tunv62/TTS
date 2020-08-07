import dlib
import cv2


class Camera:

    hog_face_detector = dlib.get_frontal_face_detector()

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 320)
        self.video.set(4, 320)

    def __int__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_hog = self.hog_face_detector(gray_img, 1)
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ret, img = cv2.imencode('.jpg', frame)
        return img.tobytes()

import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")
video = cv2.VideoCapture(0)


class SmileDetector(BoxLayout):
    def __init__(self, **kwargs):
        super(SmileDetector, self).__init__(**kwargs)
        self.img = Image()
        self.add_widget(self.img)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        check, frame = video.read()
        if not check:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Detect smiles within the detected face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            for (x1, y1, w1, h1) in smile:
                # Draw a green rectangle around the detected smile
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
                break
        self.img.texture = self.convert_frame(frame)

    def convert_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return image_texture


class SmileDetectorApp(App):
    def build(self):
        return SmileDetector()


if __name__ == '__main__':
    SmileDetectorApp().run()
    video.release()

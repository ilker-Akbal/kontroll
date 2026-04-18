import cv2

class CameraReader:
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.cap = None
        self.open()

    def open(self):
        self.close()
        self.cap = cv2.VideoCapture(self.source_url)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        return ok, frame

    def reconnect(self):
        self.open()

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
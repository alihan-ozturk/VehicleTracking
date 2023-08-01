import time
import threading
import subprocess as sp
import numpy as np
import cv2


class CustomThread(threading.Thread):
    def __init__(self, m3u8_url, cameraSize, timeSleep):
        threading.Thread.__init__(self)
        self.pipe = sp.Popen(["C:/ffmpeg/bin/ffmpeg.exe", "-i", m3u8_url,
                              "-loglevel", "quiet",
                              "-an",
                              "-f", "image2pipe",
                              "-pix_fmt", "bgr24",
                              "-vcodec", "rawvideo", "-"],
                             stdin=sp.PIPE, stdout=sp.PIPE)
        self.size = cameraSize[0] * cameraSize[1] * cameraSize[2]
        self.cameraSize = cameraSize
        self.timeSleep = timeSleep
        self.c = True
        self.lastFrame = None

    def run(self):
        while self.c:

            raw_image = self.pipe.stdout.read(self.size)
            time.sleep(self.timeSleep)
            try:
                self.lastFrame = np.frombuffer(raw_image, dtype='uint8').reshape(self.cameraSize)
            except ValueError:
                pass
        return

    def stop(self):
        self.c = False
        print("stopped")


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride) / 2, np.mod(dh, stride) / 2  # wh padding

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


class opencvThread(threading.Thread):
    def __init__(self, url, timeSleep):
        threading.Thread.__init__(self)
        self.pipe = cv2.VideoCapture(url)
        self.timeSleep = timeSleep
        self.c = True
        self.k = -5
        self.lastFrame = None

    def run(self):
        while self.c:
            ret, frame = self.pipe.read()
            if frame is not None:
                self.lastFrame = frame
            else:
                self.k += 1
                self.c = min(abs(self.k), 1)
            time.sleep(self.timeSleep)

        return

    def stop(self):
        self.c = False
        print("stopped")

import cv2 as cv
import numpy as np
import json


class click:
    FONT = cv.FONT_HERSHEY_SIMPLEX
    ALPHA = 0.5
    KEY = ord("s")

    def __init__(self, img, configName="config.txt", saveConfig=False, windowName="click"):
        self.__img = img.copy()
        self.__backup = img.copy()
        self.__windowName = windowName
        self.allPts = []

        try:
            with open(configName, 'r') as f:
                try:
                    self.allPts = json.load(f)
                    # mask = np.zeros_like(self.__img[:, :, 0])
                    mask = np.zeros(self.__img.shape[:2], dtype=np.uint8)
                    for pts in self.allPts:
                        cv.fillConvexPoly(mask, np.array(pts), 255)
                        masked = cv.bitwise_and(self.__backup, self.__backup, mask=mask)
                    cv.addWeighted(masked, self.ALPHA, self.__img, 1 - self.ALPHA, 0, self.__img)
                except json.decoder.JSONDecodeError:
                    print("file doesnt contain pts")
        except FileNotFoundError:
            print("will be create " + configName + " file")
        self.__temp = self.__img.copy()
        self.__pts = []
        self.mask = None
        self.__createMask()
        if saveConfig:
            with open(configName, 'w') as f:
                json.dump(self.allPts, f)

    def __clickEvent(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            self.__pts.append([x, y])
            cv.putText(self.__img, str(len(self.__pts)), (x, y), self.FONT,
                       1, (255, 0, 0), 2)
            cv.imshow(self.__windowName, self.__img)
        elif event == cv.EVENT_RBUTTONDOWN and len(self.__pts) > 0:
            copy = self.__temp.copy()
            del self.__pts[-1]
            for i, (x, y) in enumerate(self.__pts):
                cv.putText(copy, str(i + 1), (x, y), self.FONT,
                           1, (255, 0, 0), 2)
            self.__img = copy
            cv.imshow(self.__windowName, self.__img)

    def __createMask(self):

        cv.imshow(self.__windowName, self.__img)
        cv.setMouseCallback(self.__windowName, self.__clickEvent)
        __KEY = cv.waitKey(0)

        if len(self.__pts) > 0:
            self.allPts.append(self.__pts)
            self.__pts = []
        if len(self.allPts) == 0:
            raise Exception("no pts")
        # mask = np.zeros_like(self.__img[:, :, 0])
        mask = np.zeros(self.__img.shape[:2], dtype=np.uint8)

        self.masks = []
        for i in range(len(self.allPts)):
            cv.fillConvexPoly(mask, np.array(self.allPts[i]), i+1)
            masked = cv.bitwise_and(self.__backup, self.__backup, mask=mask)

            # mask2 = np.zeros_like(self.__img[:, :, 0])
            mask2 = np.zeros(self.__img.shape[:2], dtype=np.uint8)

            cv.fillConvexPoly(mask2, np.array(self.allPts[i]), 255)
            self.masks.append(mask2)
        if __KEY == self.KEY:
            self.__img = self.__backup.copy()
            cv.addWeighted(masked, self.ALPHA, self.__img, 1 - self.ALPHA, 0, self.__img)
            self.__temp = self.__img.copy()
            self.__createMask()
        else:
            cv.destroyWindow(self.__windowName)
            del self.__img
            del self.__backup
            del self.__temp
            self.mask = mask
            self.vis = cv.cvtColor(cv.bitwise_and(np.ones_like(mask, dtype=np.uint8)*114, cv.bitwise_not(mask)), cv.COLOR_GRAY2BGR)


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        color = colors[cat]

        label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv.rectangle(img, (x1, y1), c2, color, -1, cv.LINE_AA)
        cv.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)
    return img

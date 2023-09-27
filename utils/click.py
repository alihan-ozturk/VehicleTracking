import cv2
import numpy as np
import json


class click:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    ALPHA = 0.5
    KEY = ord("s")
    RESET = ord("r")

    def __init__(self, img, configName="config.txt", saveConfig=False, windowName="click"):
        self.img = img.copy()
        self.backup = img.copy()
        self.windowName = windowName
        self.allPts = []

        try:
            with open(configName, 'r') as f:
                try:
                    self.allPts = json.load(f)
                    mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    for pts in self.allPts:
                        cv2.fillConvexPoly(mask, np.array(pts), 255)
                        masked = cv2.bitwise_and(self.backup, self.backup, mask=mask)
                    cv2.addWeighted(masked, self.ALPHA, self.img, 1 - self.ALPHA, 0, self.img)
                except json.decoder.JSONDecodeError:
                    print("file doesnt contain pts")
        except FileNotFoundError:
            print("will be create " + configName + " file")
        self.temp = self.img.copy()
        self.pts = []
        self.mask = None
        self.createMask()
        if saveConfig:
            with open(configName, 'w') as f:
                json.dump(self.allPts, f)

    def clickEvent(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([x, y])
            cv2.putText(self.img, str(len(self.pts)), (x, y), self.FONT,
                       1, (255, 0, 0), 2)
            cv2.imshow(self.windowName, self.img)
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.pts) > 0:
            copy = self.temp.copy()
            del self.pts[-1]
            for i, (x, y) in enumerate(self.pts):
                cv2.putText(copy, str(i + 1), (x, y), self.FONT,
                           1, (255, 0, 0), 2)
            self.img = copy
            cv2.imshow(self.windowName, self.img)

    def createMask(self):

        cv2.imshow(self.windowName, self.img)
        cv2.setMouseCallback(self.windowName, self.clickEvent)
        KEY = cv2.waitKey(1)

        if len(self.pts) > 0:
            self.allPts.append(self.pts)
            self.pts = []
        if len(self.allPts) == 0:
            raise Exception("no pts")
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        self.masks = []
        for i in range(len(self.allPts)):
            cv2.fillConvexPoly(mask, np.array(self.allPts[i]), i+1)
            masked = cv2.bitwise_and(self.backup, self.backup, mask=mask)

            mask2 = np.zeros(self.img.shape[:2], dtype=np.uint8)

            cv2.fillConvexPoly(mask2, np.array(self.allPts[i]), 255)
            self.masks.append(mask2)
        if KEY == self.KEY:
            self.img = self.backup.copy()
            cv2.addWeighted(masked, self.ALPHA, self.img, 1 - self.ALPHA, 0, self.img)
            self.temp = self.img.copy()
            self.createMask()
        elif KEY == self.RESET:
            self.img = self.backup.copy()
            self.temp = self.backup.copy()
            self.pts = []
            self.allPts = []
            self.createMask()
        else:
            cv2.destroyWindow(self.windowName)
            del self.img
            del self.backup
            del self.temp
            self.mask = mask
            self.vis = cv2.cvtColor(cv2.bitwise_and(np.ones_like(mask, dtype=np.uint8)*114, cv2.bitwise_not(mask)), cv2.COLOR_GRAY2BGR)


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        color = colors[cat]

        label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img
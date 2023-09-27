import glob
import json
import shutil
import sys
import time
import cv2
import numpy as np
from utils.streamVideo import CustomThread, opencvThread, letterbox
from utils.click import click, draw_boxes
import torch
from models.experimental import attempt_load
from utils.torch_utils import TracedModel
from utils.general import non_max_suppression, scale_coords
import random
import pandas as pd
from sort import Sort

# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]
colors = [[0, 255, 223], [0, 191, 255], [80, 127, 255], [99, 49, 222]]
names = ["bus", "shuttle", "motorcycle", "car"]
classSize = len(names)
# pathLength = 128
fps = 6

model = attempt_load("./best1.pt", map_location="cuda")
model = TracedModel(model, "cuda", 800)

# m3u8_url = "https://hls.ibb.gov.tr/tkm2/hls/888.stream/chunklist.m3u8"
# streamSize = (272, 480, 3)
# fps = 10  # 12.5
# sleepTime = 1 / fps

key = ord("q")

# thread = CustomThread(m3u8_url, streamSize, sleepTime)
# thread = opencvThread(m3u8_url, sleepTime)
# thread.start()

# d = 4
# print(f"wait {d} seconds")
# time.sleep(d)

# if thread.lastFrame is not None:
#     maskC = click(thread.lastFrame, "mask1.txt", saveConfig=True)
#     startFinish = click(maskC.mask * 255 // len(maskC.masks), "startFinish1.txt", saveConfig=True)
# else:
#     thread.stop()
#     sys.exit()


for i, videoPath in enumerate(glob.glob("./records/*.mp4")):
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.15)

    # videoPath = "test2.ts"
    # videoPath = "./records/output_20230907192025.mp4"
    videoName = videoPath.split("/")[-1].split(".")[0]
    tar = videoName.replace("records", "processed") + ".avi"
    cap = cv2.VideoCapture(videoPath)
    ret, lastFrame = cap.read()



    scale_percent = 227.3  # 100
    width0 = lastFrame.shape[1]
    height0 = lastFrame.shape[0]
    width = int(width0 * scale_percent / 100)
    height = int(height0 * scale_percent / 100)
    dim = (width, height)

    writer = cv2.VideoWriter(tar,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (width0, height0))

    lastFrame = cv2.resize(lastFrame, dim, interpolation=cv2.INTER_AREA)

    print(videoName, i)
    # click(lastFrame, "./sf0.txt", saveConfig=False)

    cv2.imshow("maskChoice", lastFrame)
    if True:
        masktxt = "./mask0.txt"
        sftxt = "./sf0.txt"
        shutil.copy(masktxt, "./records/mask" + str(i) + ".txt")
        shutil.copy(masktxt, "./records/sf" + str(i) + ".txt")
    else:
        masktxt = "./records/mask" + str(i) + ".txt"
        sftxt = "./records/sf" + str(i) + ".txt"

    cv2.destroyAllWindows()

    maskC = click(lastFrame, masktxt, saveConfig=True)
    # startFinish = click(maskC.mask * 255 // len(maskC.masks), "startFinish11.txt", saveConfig=False)
    startFinish = click(lastFrame, sftxt, saveConfig=True)

    cv2.imwrite("maskss.png", startFinish.mask)

    _, thresh = cv2.threshold(maskC.mask, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh, (width0, height0), interpolation=cv2.INTER_AREA)  # resize
    maskC.mask = cv2.resize(maskC.mask, (width0, height0), interpolation=cv2.INTER_AREA)  # resize
    startFinish.mask = cv2.resize(startFinish.mask, (width0, height0), interpolation=cv2.INTER_AREA)

    map = np.zeros_like(startFinish.mask)
    pathSize = len(maskC.masks)



    temp = pd.DataFrame(columns=['id', 'class', 'startPath', 'startTime', 'cx', 'cy', 'life'])
    temp.set_index("id", inplace=True)

    history = pd.DataFrame(columns=['id', 'class', 'startPath', 'finishPath', 'speed'])
    history.set_index("id", inplace=True)
    start = 0

    while True:
        start += 1
        # im0 = thread.lastFrame.copy()
        ret, im0 = cap.read()

        if not ret:
            break
        # im0 = cv2.resize(im0, dim, interpolation=cv2.INTER_AREA) # resize
        # start = time.time()
        img = cv2.bitwise_and(im0, im0, mask=thresh)
        img[maskC.mask == 0] = 114
        img = letterbox(img)
        inputSize = img.shape
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cuda").float()
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]

        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

        for i, det in enumerate(pred):
            s = ""

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                dets_to_sort = np.empty((0, 6))
                nv = {i + 1: {j: 0 for j in range(classSize)} for i in range(pathSize)}

                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():

                    xx, yy, detclass = int((x2 - (x2 - x1) / 2)), int((y2 - (y2 - y1) / 2)), int(detclass)
                    im0[yy - 2:yy + 2, xx - 2:xx + 2] = [255, 0, 0]
                    path = maskC.mask[yy, xx]
                    if path != 0:
                        nv[path][detclass] += 1
                    label = f'{names[detclass]} {conf:.2f}'

                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                cv2.putText(im0, json.dumps(nv), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 18, 13), 2, cv2.LINE_AA)

                tracked_dets = sort_tracker.update(dets_to_sort, True)
                tracks = sort_tracker.getTrackers()

                for det in tracked_dets:
                    identities = int(det[8])
                    categories = int(det[4])
                    cx, cy = int(det[0] + (det[2] - det[0]) / 2), int(det[1] + (det[3] - det[1]) // 2)
                    sf = startFinish.mask[cy, cx]
                    map[cy, cx] = min(map[cy, cx]+20, 255)
                    if sf != 0:
                        if identities not in temp.index or sf == temp.loc[identities].startPath:
                            temp.loc[identities] = [categories, sf, start, cx, cy, 1000]
                        else:
                            arriveTime = start - temp.loc[identities].startTime
                            pixelDifference = np.sqrt(
                                (cx - temp.loc[identities].cx) ** 2 + (cy - temp.loc[identities].cy) ** 2)
                            history.loc[identities] = [categories, temp.loc[identities].startPath, sf, arriveTime]
                            temp.drop(identities, inplace=True)

                    if len(temp) > 0:
                        temp.life -= 1
                        temp = temp[temp['life'] >= 0]

                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    confidences = None

                    for t, track in enumerate(tracks):
                        track_color = colors[int(track.detclass)]
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])),
                                  (int(track.centroidarr[i + 1][0]),
                                   int(track.centroidarr[i + 1][1])),
                                  track_color, thickness=1)
                         for i, _ in enumerate(track.centroidarr)
                         if i < len(track.centroidarr) - 1]

                    im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

        cv2.imshow("frame", im0)
        writer.write(im0)
        # thread.timeSleep = max((sleepTime - (time.time() - start)), 0)
        if cv2.waitKey(1) == key:
            cv2.destroyAllWindows()
            break



    cap.release()
    writer.release()
    history.to_csv(videoName + "history.csv")
    cv2.imwrite(videoName+"map.png", map)
# thread.stop()

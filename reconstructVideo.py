import cv2
import numpy as np
import os

npySize = 1020
cap = cv2.VideoCapture("E:\\vanuatu\\vanuatu35\\640x360\\vanuatu35_%06d.jpg")
videoItr = 4

frameNum = -1

alreadyLoaded = None
probs = None

while (True):
    # Capture frame-by-frame
    frameNum += 1
    ret, frame = cap.read()
    if not ret:
        break

    npyI = (int(frameNum / npySize) + 1) * npySize
    p_path = os.path.join("E:/videoseg/VanuatuProb", "{:d}_{:05d}_P.npy".format(videoItr, npyI))
    if p_path != alreadyLoaded:
        alreadyLoaded = p_path
        probs = np.load(p_path)

    probsI = frameNum % npySize

    prob = probs[probsI, :, :, :]

    newsize = (frame.shape[1], frame.shape[0])
    prob = cv2.resize(prob, newsize)

    cv2.imshow("prob", prob)
    cv2.imshow("frame", frame)

    thresh = 0.2
    ret, mask = cv2.threshold(prob, thresh, 1.0, cv2.THRESH_BINARY)
    ret, moreCertainMask = cv2.threshold(prob, 0.5, 1.0, cv2.THRESH_BINARY)

    _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, moreCertainContours, hierarchy = cv2.findContours(moreCertainMask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    notThreshedI = np.where(mask==0)
    mask[notThreshedI] = prob[notThreshedI] * (1/thresh)

    frame[:, :, 0] = frame[:, :, 0] * mask
    frame[:, :, 1] = frame[:, :, 1] * mask
    frame[:, :, 2] = frame[:, :, 2] * mask

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
    cv2.drawContours(frame, moreCertainContours, -1, (255, 0, 0), 1)

    cv2.imshow("frameR", frame)
    cv2.waitKey(1)

cap.release()
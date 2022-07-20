# todo compare same images

import cv2

cap = cv2.VideoCapture('XYZ.avi')
# For streams:
#   cap = cv2.VideoCapture('rtsp://url.to.stream/media.amqp')
# Or e.g. most common ID for webcams:
#   cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        count += 100
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        cap.release()
        break
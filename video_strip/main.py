# todo compare same images

import cv2
from skimage.metrics import structural_similarity as ssim

cap = cv2.VideoCapture('XYZ.avi')
# For streams:
#   cap = cv2.VideoCapture('rtsp://url.to.stream/media.amqp')
# Or e.g. most common ID for webcams:
#   cap = cv2.VideoCapture(0)
count = 0

def compare_img(img1, img2):
    original = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    test = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if ssim(original, test) < 0.8:
        return False
    else:
        return True


while cap.isOpened():
    ret, frame = cap.read()
    prev_frame = None
    if ret:
        if not compare_img(frame, prev_frame):
            cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        count += 100
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        cap.release()
        break

import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


# for rescaling
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# make_1080p()
# change_res(2000, 1000)

while True:
    # Capture frame by frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=30)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frame2 = rescale_frame(frame, percent=140)
    cv2.imshow('frame2', frame2)
    frame3 = rescale_frame(frame, percent=240)
    cv2.imshow('frame3', frame3)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release capture when everything is done
cap.release()
cv2.destroyAllWindows()

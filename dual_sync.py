import numpy as np
import cv2
import glob
import argparse
import sys

def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Set the values for your cameras


capL = cv2.VideoCapture(gstreamer_pipeline(
            sensor_id=0,
            sensor_mode=3,
            flip_method=2,
            display_height=540,
            display_width=960,
        ),cv2.CAP_GSTREAMER)
capR = cv2.VideoCapture(gstreamer_pipeline(
            sensor_id=1,
            sensor_mode=3,
            flip_method=2,
            display_height=540,
            display_width=960,
        ),cv2.CAP_GSTREAMER)


# Use these if you need high resolution.
# capL.set(3, 1024) # width
# capL.set(4, 768) # height

# capR.set(3, 1024) # width
# capR.set(4, 768) # height
i = 0


def main():
    global i
    if len(sys.argv) < 3:
        print("Usage: ./program_name directory_to_save start_index")
        sys.exit(1)

    i = int(sys.argv[2])  # Get the start number.

    while True:
        # Grab and retreive for sync
        if not (capL.grab() and capR.grab()):
            print("No more frames")
            break

        _, leftFrame = capL.retrieve()
        _, rightFrame = capR.retrieve()

        # Use if you need high resolution. If you set the camera for high res, you can pass these.
        cv2.namedWindow('capL', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('capR', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('capL', leftFrame)
        cv2.imshow('capR', rightFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(sys.argv[1] + "/left" + str(i) + ".png", leftFrame)
            cv2.imwrite(sys.argv[1] + "/right" + str(i) + ".png", rightFrame)
            i += 1

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
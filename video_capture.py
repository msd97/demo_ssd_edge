import cv2 
import os
import logging
import tensorflow as tf
import tensorflow_hub as hub
from ssd_tf_hub import inference_hub

logging.basicConfig(format="%(asctime)s // %(levelname)s : %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

def gstreamer_pipeline(
    sensor_id=0,
    #sensor_mode=3,
    capture_width=3280,
    capture_height=2464,
    display_width=816,
    display_height=616,
    framerate=21/1,
    flip_method=2,
):
    return (
       "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
        % (
            sensor_id,
            #sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


os.environ['TFHUB_CACHE_DIR'] = './hub_directory/tf_cache'

m_path = 'ssd_mobilenet_v2_2_trt'
im_size = (224,224)

model = tf.saved_model.load(m_path)
logging.info('Model successfully loaded')

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture(0)

ret,frame = cap.read()
cv2.imshow('frame', frame)

logging.info('Video streamer initialized')
  
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()

    detection = inference_hub(frame, model)
    #detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
    # Display the resulting frame
    cv2.imshow('frame', detection)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
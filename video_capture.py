import cv2 
import os
import tensorflow as tf
import tensorflow_hub as hub
from ssd_tf_hub import inference

cap = cv2.VideoCapture(0)

os.environ['TFHUB_CACHE_DIR'] = './hub_directory/tf_cache'

m_path = 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'
im_size = (224,224)

model = hub.load(m_path).signatures['default']
  
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()

    detection = inference(frame, model)
    detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
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
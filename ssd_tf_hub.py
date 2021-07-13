import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

def draw_boxes(img, boxes, class_names, scores):

    max_boxes = 10
    min_score = 0.5

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            H= img.shape[0]
            W = img.shape[1]

            ymin, xmin, ymax, xmax = tuple(boxes[i].numpy())
            xmin = int(W * xmin)
            ymin = int(H * ymin)
            xmax = int(W * xmax)
            ymax = int(H * ymax)

            label = class_names[i].numpy().decode('ascii')
            mark = label + " {:.2f}%".format(100*scores[i].numpy())

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            img = cv2.putText(img, mark, (xmin, ymin-11), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    return img


#tf.compat.v1.disable_eager_execution()
os.environ['TFHUB_CACHE_DIR'] = './hub_directory/tf_cache'

m_path = 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'
im_size = (224,224)
im_path = 'dogs.jpg'

model = hub.load(m_path).signatures['default']

test_im = cv2.imread(im_path)
test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
resized= cv2.resize(test_im, im_size)
x = np.array(resized)
x = np.expand_dims(x, axis=0)

converted_img  = tf.image.convert_image_dtype(resized, tf.float32)[tf.newaxis, ...]
pred = model(converted_img)
boxes = pred['detection_boxes']
print(tuple(boxes[0].numpy()))
#print(pred)

marked = draw_boxes(test_im, pred['detection_boxes'], pred['detection_class_entities'], pred['detection_scores'])

test_im = cv2.imread(im_path)
test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1,2, figsize=(200,200))
ax[0].imshow(test_im)
ax[0].set_title('Raw image')
ax[1].imshow(marked)
ax[1].set_title('Marked image')
plt.show()
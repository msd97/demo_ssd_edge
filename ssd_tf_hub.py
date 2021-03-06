import logging
import cv2
import numpy as np
import tensorflow as tf

def draw_boxes(img, boxes, class_names, scores):

    max_boxes = 10
    min_score = 0.6

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            H= img.shape[0]
            W = img.shape[1]

            ymin, xmin, ymax, xmax = tuple(boxes[i])
            xmin = int(W * xmin)
            ymin = int(H * ymin)
            xmax = int(W * xmax)
            ymax = int(H * ymax)

            label = str(class_names[i])
            mark = label + " {:.2f}%".format(100*scores[i])
            logging.info('Obtained predictions for frame: {}'.format(mark))

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            img = cv2.putText(img, mark, (xmin, ymin-11), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    return img


def inference_hub(img, model):

    input_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    input_im = np.array(input_im)
    input_im = np.expand_dims(input_im, axis=0)
    input_im = tf.cast(input_im, tf.uint8)

    logging.info('Frame preprocessed, now performing inference')
    pred = model(input_im)

    boxes = np.squeeze(pred['detection_boxes'].numpy())
    labels = np.squeeze(pred['detection_classes'].numpy())
    scores = np.squeeze(pred['detection_scores'].numpy())
    marked = draw_boxes(img, boxes, labels, scores)

    return marked

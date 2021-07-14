import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

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


def inference(img, model):

    recolored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    converted_img  = tf.image.convert_image_dtype(recolored, tf.float32)[tf.newaxis, ...]
    pred = model(converted_img)

    marked = draw_boxes(recolored, pred['detection_boxes'], pred['detection_class_entities'], pred['detection_scores'])

    return marked
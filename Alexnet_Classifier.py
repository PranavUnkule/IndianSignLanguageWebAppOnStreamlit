import cv2
import numpy as np
import tensorflow as tf


def load_alexnet():
    model_dir = 'Models//AlexNet.json'
    model_weights_dir = 'Models//AlexNet_Weights.hdf5'
    with open(model_dir, 'r') as json_file:
        json_saved_model = json_file.read()
    model = tf.keras.models.model_from_json(json_saved_model)
    model.load_weights(model_weights_dir)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


def frame_processing(frame):
    t, r, b, l = 100, 350, 325, 575  # size of window
    frame = cv2.flip(frame, 1)
    roi = frame[140:380, 235:440]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lh = 0
    ls = 0
    lv = 0
    uh = 255
    us = 32
    uv = 255
    l_b = np.array([lh, ls, lv])
    u_b = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, l_b, u_b)
    mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(roi, roi, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    (cnts, _) = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        # Storing the max contors (Which area is maximum)
        max_cont = max(cnts, key=cv2.contourArea)
        if max_cont is not None:
            mask = cv2.drawContours(res, [max_cont + (r, t)], -1, (0, 0, 255))
            mask = np.zeros(res.shape, dtype="uint8")
            cv2.drawContours(mask, [max_cont], -1, 255, -1)

            res = cv2.bitwise_and(res, res, mask=mask)

            high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lowThresh = 0.5 * high_thresh
            res = cv2.Canny(res, lowThresh, high_thresh)
            cv2.imshow('Canny', res)
            # Prediction
            if res is not None and cv2.contourArea(max_cont) > 1000:
                final_res = cv2.resize(res, (224, 224))
                final_res = final_res / 255
                final_res = final_res.reshape(-1, 224, 224, 1)

    return res, cv2.contourArea(max_cont), final_res


def predict_alexnet(model, frame):
    signs = visual_dict={0:'1',1:'2',2:'3',3:'A',4:'B',5:'C',6:'J',7:'My',8:'Name',9:'Y'}
    res, maxcnts, frame_processed = frame_processing(frame)
    if res is not None and maxcnts > 1000:
        output = model.predict(frame_processed)
        prob = np.amax(output)
        sign = np.argmax(output)
        prediction = signs[sign]
    else:
        prediction = ''
    return prediction

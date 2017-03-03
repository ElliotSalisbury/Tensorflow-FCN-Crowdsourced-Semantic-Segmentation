#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn16_vgg
import fcn32_vgg
import utils
from loss import loss
import cv2
import random

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

img1 = cv2.imread("./test_data/tabby_cat.png")
NUM_CLASSES = 2

def run_network():
    with tf.Session() as sess:
        images = tf.placeholder("float")
        batch_images = tf.expand_dims(images, 0)

        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(batch_images, num_classes=NUM_CLASSES, random_init_fc8=False, train=True, debug=True)

        print('Finished building Network.')

        logging.warning("Score weights are initialized random.")
        logging.warning("Do not expect meaningful results.")


        print("Build Training network")
        annotations = tf.placeholder("float")
        # batch_annotations = tf.expand_dims(annotations, 0)

        vgg_fcn_loss = loss(vgg_fcn.upscore, annotations, NUM_CLASSES)
        optimizer = tf.train.AdamOptimizer(1e-6)
        train_step = optimizer.minimize(vgg_fcn_loss)

        logging.info("Start Initializing Variabels.")
        init = tf.global_variables_initializer()
        sess.run(init)

        print("Training the Network")
        saver = tf.train.Saver()

        Xs = []
        Ys = []
        for i in range(305):
            f_path = os.path.join("E:/videoseg/MyGT", "{:05d}_F.png".format(i))
            gt_path = os.path.join("E:/videoseg/MyGT", "{:05d}_GT.png".format(i))
            frame = cv2.imread(f_path)
            gt = cv2.imread(gt_path)

            # h, w, d = frame.shape
            # newsize = (int(w / 2), int(h / 2))
            # frame = cv2.resize(frame, newsize)
            # gt = cv2.resize(gt, newsize)

            mask = np.zeros((gt.shape[0], gt.shape[1], NUM_CLASSES), dtype=np.float32)
            mask[:, :, 0] = 1

            color_classes = [(128, 64, 128), ]
            for i, color in enumerate(color_classes):
                isColor = np.logical_and(np.logical_and(gt[:, :, 0] == color[0], gt[:, :, 1] == color[1]), gt[:, :, 2] == color[2])
                isColorI = np.argwhere(isColor)

                value = np.zeros((NUM_CLASSES), dtype=np.float32)
                value[i + 1] = 1.0

                mask[isColorI[:, 0], isColorI[:, 1], :] = value

            Xs.append(frame)
            Ys.append(mask)

        for i in range(501):
            # if i%50 == 0:
            print("step {}".format(i))
            X = random.sample(Xs, 1)[0]
            Y = random.sample(Ys, 1)[0]
            train_step.run(feed_dict={images: X, annotations: Y})

            if i%100 == 0:
                save_path = saver.save(sess, "./chkpt/model32.ckpt")
                print("Model saved in file: %s" % save_path)

        # saver.restore(sess, "./chkpt/model.ckpt")

        print('Running the Network')
        # feed_dict = {images: img1}
        # tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
        # down, up = sess.run(tensors, feed_dict=feed_dict)
        #
        # down_color = utils.color_image(down[0])
        # up_color = utils.color_image(up[0])

        # down_color_up = cv2.resize(down_color[:, :, :3], (img1.shape[1], img1.shape[0]))
        # cv2.imshow("up", ((img1 * 0.5) + (down_color_up * (255 * 0.5))).astype(np.uint8))
        # cv2.waitKey(-1)
        # scp.misc.imsave('fcn16_downsampled.png', down_color)
        # scp.misc.imsave('fcn16_upsampled.png', up_color)

        # Loading images
        cap = cv2.VideoCapture(r"E:\videoseg\0016E5.MXF")
        frameNum = -2
        while (True):
            # Capture frame-by-frame
            frameNum += 1
            for i in range(3):
                ret, frame = cap.read()
            if not ret:
                break

            #process frame
            h,w,d = frame.shape
            newsize = (int(w/2),int(h/2))
            frame = cv2.resize(frame,newsize)

            feed_dict = {images: frame}
            tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
            down,up = sess.run(tensors, feed_dict=feed_dict)

            down_color = utils.color_image(down[0], num_classes=NUM_CLASSES)
            up_color = utils.color_image(up[0], num_classes=NUM_CLASSES)

            down_color_up = cv2.resize(down_color.squeeze()[:,:,:3], newsize)

            cv2.imshow("down", ((frame * 0.5) + (down_color_up * (255 * 0.5))).astype(np.uint8))
            cv2.imshow("up", ((frame * 0.5) + (up_color[:,:,:3] * (255 * 0.5))).astype(np.uint8))
            cv2.waitKey(1)

        cap.release()

if __name__ == "__main__":
    run_network()
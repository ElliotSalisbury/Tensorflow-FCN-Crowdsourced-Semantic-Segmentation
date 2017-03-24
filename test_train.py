#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys
import datetime
import fcn16_vgg
import fcn32_vgg
import utils
from loss import loss
import cv2
import random
import pickle
from BatchDatasetReader2 import BatchDataset2

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

img1 = cv2.imread("./test_data/tabby_cat.png")
NUM_CLASSES = 2

def load_data(data_dir):
    pickle_filename = "filelist.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):

        datatypes = ['training', 'validation']
        image_list = {'training':[], 'validation':[]}

        for i in range(11941):
            f_path = os.path.join("E:/videoseg/VanuatuGT", "{:05d}_F.png".format(i))
            gt_path = os.path.join("E:/videoseg/VanuatuGT", "{:05d}_GT.png".format(i))
            if not os.path.exists(f_path):
                continue

            type = 'training'
            if i % 100 == 0:
                type = 'validation'

            #create results
            record = {'image': f_path, 'annotation': gt_path, 'frameId':i}
            image_list[type].append(record)

        for type in image_list:
            random.shuffle(image_list[type])

        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(image_list, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records

def run_network():
    with tf.Session() as sess:
        images = tf.placeholder("float")
        # batch_images = tf.expand_dims(images, 0)

        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

        vgg_fcn = fcn32_vgg.FCN32VGG()
        with tf.name_scope("content_vgg"):
            vgg_fcn.build(images, num_classes=NUM_CLASSES, train=True, debug=True, keep_prob=keep_probability)

        print('Finished building Network.')

        logging.warning("Score weights are initialized random.")
        logging.warning("Do not expect meaningful results.")


        print("Build Training network")
        annotations = tf.placeholder("float")
        # batch_annotations = tf.expand_dims(annotations, 0)

        vgg_fcn_loss = loss(vgg_fcn.upscore, annotations, NUM_CLASSES)
        tf.summary.scalar("entropy", vgg_fcn_loss)

        trainable_vars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(1e-6)
        grads = optimizer.compute_gradients(vgg_fcn_loss, var_list=trainable_vars)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradient/" + var.op.name, grad)
        train_step = optimizer.apply_gradients(grads)


        pred2color = int(255.0/(NUM_CLASSES-1.0))
        CLASS_WE_CARE_ABOUT = 1
        tf.summary.image("input_image", images, max_outputs=1)
        tf.summary.image("ground_truth", tf.cast(tf.split(annotations, NUM_CLASSES, axis=3)[CLASS_WE_CARE_ABOUT], tf.uint8)*pred2color, max_outputs=1)
        tf.summary.image("pred_annotation", tf.cast((tf.split(vgg_fcn.prob_up, NUM_CLASSES, axis=3)[CLASS_WE_CARE_ABOUT])*pred2color, tf.uint8), max_outputs=1)

        logging.info("Start Initializing Variables.")
        init = tf.global_variables_initializer()
        sess.run(init)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./chkpt/", sess.graph)

        sum_valid_image = tf.summary.image("valid_input_image", images, max_outputs=20)
        sum_valid_gt = tf.summary.image("valid_ground_truth", tf.cast(tf.split(annotations, NUM_CLASSES, axis=3)[1], tf.uint8) * pred2color,
                         max_outputs=20)
        sum_valid_pred = tf.summary.image("valid_pred_annotation",
                         tf.cast((tf.split(vgg_fcn.prob_up, NUM_CLASSES, axis=3)[CLASS_WE_CARE_ABOUT]) * pred2color, tf.uint8), max_outputs=20)
        sum_valid_entropy = tf.summary.scalar("valid_entropy", vgg_fcn_loss)

        valid_summary_op = tf.summary.merge([sum_valid_image,sum_valid_gt,sum_valid_pred,sum_valid_entropy])

        training_records, validation_records = load_data("E:/videoseg/VanuatuGT")

        image_options = {'resize': True, 'resize_size': 244}
        DURATION = 30 * 60 * 2
        train_dataset_reader = BatchDataset2(training_records, NUM_CLASSES, image_options, fromFrameId=0, uptoFrameId=DURATION)
        validation_dataset_reader = BatchDataset2(validation_records, NUM_CLASSES, image_options)

        print("Training the Network")
        saver = tf.train.Saver()
        # saver.restore(sess, "./chkpt/model32.ckpt")
        MAX_ITERATION = int(500)
        VIDEO_STEPS = int(5)
        BATCH_SIZE = 1
        for itr in range(1, MAX_ITERATION*VIDEO_STEPS + 2):
            train_images, train_annotations = train_dataset_reader.next_batch(BATCH_SIZE)
            feed_dict = {images: train_images, annotations: train_annotations, keep_probability: 0.4}

            train_step.run(feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([vgg_fcn_loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

                valid_images, valid_annotations = validation_dataset_reader.next_batch(20)
                feed_dict = {images: valid_images, annotations: valid_annotations, keep_probability: 1.0}
                valid_loss, valid_summary_str = sess.run([vgg_fcn_loss, valid_summary_op], feed_dict=feed_dict)
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                summary_writer.add_summary(valid_summary_str, itr)
            if itr % 500 == 0:
                save_path = saver.save(sess, "./chkpt/model32.ckpt")
                print("Model saved in file: %s" % save_path)

            if itr % MAX_ITERATION == 0:
                #we've finished training for this portion of the video, lets save out the video and start training on the next part of the video
                videoItr = int(itr / MAX_ITERATION)

                print("processing the entire video")
                cap = cv2.VideoCapture("E:\\vanuatu\\vanuatu35\\640x360\\vanuatu35_%06d.jpg")
                frameNum = 0
                allFrames = None
                while (True):
                    # Capture frame-by-frame
                    frames = []
                    ret = False
                    for i in range(20):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frameNum += 1

                        frame = cv2.resize(frame, (image_options['resize_size'], image_options['resize_size']))

                        if frame is not None and frame.any():
                            frames.append(frame)

                    if frames:
                        feed_dict = {images: np.array(frames), keep_probability: 1.0}
                        prob = sess.run([vgg_fcn.prob_up], feed_dict=feed_dict)
                        prob = prob[0][:, :, :, 1, None]

                        if allFrames is None:
                            allFrames = prob
                        else:
                            allFrames = np.vstack((allFrames, prob))

                    if allFrames.shape[0] > 1000 or not ret:
                        print("saving frames {} - {}".format(frameNum-allFrames.shape[0], frameNum))
                        p_path = os.path.join("E:/videoseg/VanuatuProb", "{:d}_{:05d}_P.npy".format(videoItr, frameNum))
                        np.save(p_path, allFrames)
                        allFrames = None

                    if not ret:
                        break
                cap.release()

                startFrameId = int(videoItr * 30 * 60)
                print("loading the next section of the training data. {} - {}".format(startFrameId,
                                                                                      startFrameId + DURATION))
                train_dataset_reader.changeFrameIdRange(fromFrameId=startFrameId, uptoFrameId=startFrameId+DURATION)


if __name__ == "__main__":
    run_network()
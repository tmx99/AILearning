# -*- coding: utf-8 -*-
"""
Created on 2020/0730

@author: Jun Zhang
"""

import os
import tensorflow as tf
import numpy as np
import read_data
from sklearn.metrics import roc_auc_score, auc, recall_score, accuracy_score
from tensorflow.python import pywrap_tensorflow

class Model():
    def __init__(self, sess, seed, save_dir, x, y_, keep_prob):
        self.seed = seed
        self.x = x
        self.y_ = y_
        self.sess = sess
        self.keep_prob = keep_prob
        self.save_dir = save_dir
        self.y_conv = self.define_cnn_model(self.x, self.keep_prob)
        self.cross_entropy = self.my_loss(self.y_, self.y_conv)
        # train_vars = tf.get_collection('train_vars')
        # self.step = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy)
        self.saver = tf.train.Saver(max_to_keep=10)

    def weight_variable(self, shape):
        '''
        initialize weight variables
        :param shape: shape of weight variables
        :return:
        '''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        '''
        initialize bias variables
        :param shape: shape of bias variables
        :return:
        '''
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        '''
        convolution operation
        :param x: input
        :param W: weight variables
        :return:
        '''
        return tf.nn.conv2d(x, w, strides=[1, 1, 20, 1], padding='SAME')

    def my_loss(self, y, y_p):
        '''
        loss function used in this project
        :param y: true proteins labels
        :param y_p: predicted scores
        :return:
        '''
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        y_d = y[:, 0]
        y_r = y[:, 1]
        y_pd = y_p[:, 0]
        y_pr = y_p[:, 1]
        loss1 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_d, logits=y_pd, pos_weight=1.5))
        loss2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_r, logits=y_pr, pos_weight=1.5))
        return (loss1 + loss2)/2.0 + reg

    def my_auc_score(self, y, y_p):
        '''
        calculate subset accuracy for multi-label prediction
        :param y: an array, true proteins labels
        :param y_p: an array, predicted scores
        :return:
        '''
        y = y[:, 1].tolist()
        y_p = y_p[:, 1].tolist()

        auc = roc_auc_score(y, y_p)
        for i in range(len(y_p)):
            if (y_p[i] > 0.5):
                y_p[i] = 1
            else:
                y_p[i] = 0
        ACC = accuracy_score(y, y_p)
        return auc, ACC


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.allow_growth = True
    tf.set_random_seed(231)
    data = read_data.Input()

    save_dir = '../results/FirstLayer'
    pssm_dir = '../data/'
    seq_dir = '../data/'
    batch_size = 32
    seed = 231

    #Ô¤²â¹ý³Ì
    with tf.Session(config=config) as sess:
        seq_test_dir = '../data/'

        mats, labels, names = data.get_pssm_varDic_2l(seq_test_dir + 'AAPT.txt',
                                                    pssm_dir)

        model_path = '../results/FirstLayer/tf_model-100.meta'
        weight_path = '../results/FirstLayer/tf_model-100'
        pred_scores = []
        name_list = []

        reader = pywrap_tensorflow.NewCheckpointReader(weight_path) 
        vars = reader.get_variable_to_shape_map()

        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, weight_path)
        
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        y = tf.get_collection('predict')[0]
        
        ouputs = sess.run(y, feed_dict={x: mats, keep_prob: 1.0})
        pred_scores.append(ouputs)

        name_list.extend(names)

        pred_scores = np.vstack(pred_scores)
        pred_y = sess.run(tf.nn.softmax(pred_scores))

        pred_label = []
        
        for i in pred_y:
            pred_label.append(i[1])
        return pred_label
        with open('../results/FirstLayer/predicted_scores.txt', 'w') as f:
            f.write('pred auc\ttrue auc\n')
            for i in range(len(ture_y)):
                f.write('%.16f\t%.1f\n' % (pred_y[i][1], ture_y[i][1]))


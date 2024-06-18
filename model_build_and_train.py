# -*- coding: utf-8 -*-
"""
Created on 2020/07/30

@author: Jun Zhang
"""
import argparse
import os
import tensorflow as tf
import numpy as np
import read_data
import split_data
import time
from sklearn.metrics import roc_auc_score

class Model():
    def __init__(self, sess, seed, save_dir, x, y_, keep_prob):
        self.seed = seed
        self.x = x
        self.y_ = y_
        self.sess = sess
        self.keep_prob = keep_prob
        self.save_dir = save_dir
        self.y_conv = self.define_MotifCNN_model(self.x, self.keep_prob)
        self.cross_entropy = self.my_loss(self.y_, self.y_conv)
        train_vars = tf.get_collection('train_vars')
        self.step = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy, var_list=train_vars)
        self.saver = tf.train.Saver(max_to_keep=10)


    def read_motif(self, path):
        MOTIFS = []
        PSSM_ALP = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                'P', 'S', 'T', 'W', 'Y', 'V']

        Motif_ALP = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                'R', 'S', 'T', 'W', 'Y', 'V']

        with open(path, 'r') as f:
            lines = f.readlines()
            i = 7
            while i >= 0:
                if i >= len(lines):
                    break
                elif len(lines[i].split()) < 1:
                    i += 1
                elif lines[i].split()[0] == 'MOTIF':
                    tmp = []
                    i += 3
                elif lines[i].split()[0] == 'URL':
                    MOTIFS.append(tmp)
                    i += 2
                else:
                    tmp.append(lines[i].split())
                    i += 1
        M = []
        for motif in MOTIFS:
            tmp = np.asarray(motif, dtype=np.float32)
            for i in range(tmp.shape[0]):
                for j in range(20):
                    ti = PSSM_ALP.index(Motif_ALP[j])
                    tmp[i][ti] = float(motif[i][j])
            M.append(tmp)
        return M

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
        y_d = y[:, 0]
        y_r = y[:, 1]
        y_pd = y_p[:, 0]
        y_pr = y_p[:, 1]
        dna_auc = roc_auc_score(y_d[:, 0], y_pd[:, 0])
        rna_auc = roc_auc_score(y_r[:, 0], y_pr[:, 0])
        return dna_auc, rna_auc

    def weight_variable(self, shape):
        '''
        initialize weight variables
        :param shape: shape of weight variables
        :return:
        '''
        # 截断正态分布
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

    def define_cnn_model(self, x, keep_prob):
        '''
        construct a CNN network
        :param x: input
        :param keep_prob: probability of dropout
        :return:
        '''
        # 管理传给get_variable()的变量名称的作用域，得到四个张量
        with tf.variable_scope("conv1"):
            w_conv1 = self.weight_variable([5, 20, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(x, w_conv1) + b_conv1, name='conv1_output')
            m_pool11 = tf.reduce_max(h_conv1, axis=1)

        with tf.variable_scope("conv2"):
            w_conv2 = self.weight_variable([7, 20, 1, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(x, w_conv2) + b_conv2, name='conv2_output')
            m_pool21 = tf.reduce_max(h_conv2, axis=1)

        with tf.variable_scope("conv3"):
            w_conv3 = self.weight_variable([10, 20, 1, 36])
            b_conv3 = self.bias_variable([36])
            h_conv3 = tf.nn.relu(self.conv2d(x, w_conv3) + b_conv3, name='conv3_output')
            m_pool31 = tf.reduce_max(h_conv3, axis=1)

        with tf.variable_scope("conv4"):
            w_conv4 = self.weight_variable([20, 20, 1, 32])
            b_conv4 = self.bias_variable([32])
            h_conv4 = tf.nn.relu(self.conv2d(x, w_conv4) + b_conv4, name='conv4_output')
            m_pool41 = tf.reduce_max(h_conv4, axis=1)

        with tf.variable_scope("dense"):
            h_concat1 = tf.concat([m_pool11, m_pool21, m_pool31, m_pool41], -1) # 拼接张量
            h_concat1_flat = tf.reshape(h_concat1, [-1,  164])
            w_fc1 = self.weight_variable([164, 128])
            b_fc1 = self.bias_variable([128])
            h_fc1 = tf.nn.relu(tf.matmul(h_concat1_flat, w_fc1) + b_fc1, name='dense')

        with tf.variable_scope("output"):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #用在全连接层，防止过拟合
            w_fc2 = self.weight_variable([128, 4])
            b_fc2 = self.bias_variable([4])
            pred_scores = tf.matmul(h_fc1_drop, w_fc2) + b_fc2 # 矩阵相乘相加
            output = tf.reshape(pred_scores, [-1, 2, 2])
        return output

    def define_motif_model(self, x, keep_prob):
        '''
        construct a CNN network
        :param x: input
        :param keep_prob: probability of dropout
        :return:
        '''

        motif_feature = []
        n = 0
        with tf.variable_scope("motif_conv1"):
            motifs = self.read_motif('./data/mega-motif467.meme')
            for motif in motifs:
                n += 1
                w_value = tf.constant(motif)
                w_value = tf.reshape(w_value, [len(motif), 20, 1, 1])
                b_value = self.bias_variable([1])
                tf.add_to_collection("train_vars", b_value)
                w_motif_conv = tf.Variable(w_value)
                b_motif_conv = tf.Variable(b_value)
                h_motif_conv = tf.nn.relu(self.conv2d(x, w_motif_conv) + b_motif_conv, name='motif' + str(n))
                motif_max = tf.reduce_max(h_motif_conv, axis=1)
                motif_feature.append(motif_max)
        print('mega:', n)

        with tf.variable_scope("dense"):
            h_concat1 = tf.concat(motif_feature, -1)
            h_concat1_flat = tf.reshape(h_concat1, [-1, n])
            w_fc1 = self.weight_variable([n, 128])
            b_fc1 = self.bias_variable([128])
            tf.add_to_collection("train_vars", w_fc1)
            tf.add_to_collection("train_vars", b_fc1)
            h_fc1 = tf.nn.relu(tf.matmul(h_concat1_flat, w_fc1) + b_fc1, name='final_features')

        with tf.variable_scope("output"):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            w_fc2 = self.weight_variable([128, 4])
            b_fc2 = self.bias_variable([4])
            tf.add_to_collection("train_vars", w_fc2)
            tf.add_to_collection("train_vars", b_fc2)
            pred_scores = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
            output = tf.reshape(pred_scores, [-1, 2, 2])
        return output

    def define_MotifCNN_model(self, x, keep_prob):
        '''
        construct a MotifCNN network
        :param x: input
        :param keep_prob: probability of dropout
        :return:
        '''

        motif_feature = []
        with tf.variable_scope("conv1"):
            w_conv1 = self.weight_variable([5, 20, 1, 32])
            b_conv1 = self.bias_variable([32])
            tf.add_to_collection("train_vars", w_conv1)
            tf.add_to_collection("train_vars", b_conv1)
            h_conv1 = tf.nn.relu(self.conv2d(x, w_conv1) + b_conv1, name='conv1_output')
            m_pool11 = tf.reduce_max(h_conv1, axis=1)
            motif_feature.append(m_pool11)

        with tf.variable_scope("conv2"):
            w_conv2 = self.weight_variable([7, 20, 1, 64])
            b_conv2 = self.bias_variable([64])
            tf.add_to_collection("train_vars", w_conv2)
            tf.add_to_collection("train_vars", b_conv2)
            h_conv2 = tf.nn.relu(self.conv2d(x, w_conv2) + b_conv2, name='conv2_output')
            m_pool21 = tf.reduce_max(h_conv2, axis=1)
            motif_feature.append(m_pool21)

        with tf.variable_scope("conv3"):
            w_conv3 = self.weight_variable([10, 20, 1, 36])
            b_conv3 = self.bias_variable([36])
            tf.add_to_collection("train_vars", w_conv3)
            tf.add_to_collection("train_vars", b_conv3)
            h_conv3 = tf.nn.relu(self.conv2d(x, w_conv3) + b_conv3, name='conv3_output')
            m_pool31 = tf.reduce_max(h_conv3, axis=1)
            motif_feature.append(m_pool31)

        with tf.variable_scope("conv4"):
            w_conv4 = self.weight_variable([20, 20, 1, 32])
            b_conv4 = self.bias_variable([32])
            tf.add_to_collection("train_vars", w_conv4)
            tf.add_to_collection("train_vars", b_conv4)
            h_conv4 = tf.nn.relu(self.conv2d(x, w_conv4) + b_conv4, name='conv4_output')
            m_pool41 = tf.reduce_max(h_conv4, axis=1)
            motif_feature.append(m_pool41)

        n=0
        with tf.variable_scope("motif_conv2"):
            motifs = self.read_motif('data/motif/mega-motif467.meme')
            for motif in motifs:
                n += 1
                w_value = tf.constant(motif)
                w_value = tf.reshape(w_value, [len(motif), 20, 1, 1])
                b_value = self.bias_variable([1])
                tf.add_to_collection("train_vars", b_value)
                w_motif_conv = tf.Variable(w_value, )
                b_motif_conv = tf.Variable(b_value)
                h_motif_conv = tf.nn.relu(self.conv2d(x, w_motif_conv) + b_motif_conv, name='motif' + str(n))
                motif_max = tf.reduce_max(h_motif_conv, axis=1)
                motif_feature.append(motif_max)

        with tf.variable_scope("dense1"):
            h_concat1 = tf.concat(motif_feature, -1)
            h_concat1_flat = tf.reshape(h_concat1, [-1, 631])
            w_fc1 = self.weight_variable([631, 128])
            b_fc1 = self.bias_variable([128])
            tf.add_to_collection("train_vars", w_fc1)
            tf.add_to_collection("train_vars", b_fc1)
            h_fc1 = tf.nn.relu(tf.matmul(h_concat1_flat, w_fc1) + b_fc1, name='final_features')

        with tf.variable_scope("output"):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            w_fc2 = self.weight_variable([128, 4])
            b_fc2 = self.bias_variable([4])
            tf.add_to_collection("train_vars", w_fc2)
            tf.add_to_collection("train_vars", b_fc2)
            pred_scores = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
            output = tf.reshape(pred_scores, [-1, 2, 2])
        return output

    def train(self, train_x, train_y, val_x, val_y, epochs, batch_size):
        '''
        train prediction model
        :param train_x:
        :param train_y:
        :param epochs:
        :param batch_size:
        :return:
        '''
        # 向当前计量图中添加张量集合
        tf.add_to_collection("predict", self.y_conv)
        keys = list(train_x.keys())
        init = tf.global_variables_initializer()
        self.sess.run(init)
        np.random.seed(self.seed)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        with open(self.save_dir + '/loss.txt', 'a') as f:
            f.write('epoch\ttrain DAUC\ttrain RAUC\ttrain loss\tval DAUC\tval RAUC\tval loss\n')
            if val_x is not None:
                pred_scores = []
                ture_labels = []
                for k in val_x.keys():
                    score = self.y_conv.eval(feed_dict={self.x: val_x[k],
                                                        self.keep_prob: 1.})
                    pred_scores.append(score)
                    ture_labels.append(val_y[k])
                pred_y = np.vstack(pred_scores)
                ture_y = np.vstack(ture_labels)
                pred_sy = self.sess.run(tf.nn.softmax(pred_y))
                val_dauc, val_rauc = self.my_auc_score(ture_y, pred_sy)
                val_loss = self.sess.run(self.my_loss(ture_y, pred_y))
            train_pred_scores = []
            train_ture_labels = []
            for k in train_x.keys():
                train_score = self.y_conv.eval(feed_dict={self.x: train_x[k],
                                                          self.keep_prob: 1.})
                train_pred_scores.append(train_score)
                train_ture_labels.append(train_y[k])
            train_pred_y = np.vstack(train_pred_scores)
            train_ture_y = np.vstack(train_ture_labels)
            train_pred_sy = self.sess.run(tf.nn.softmax(train_pred_y))
            train_dauc, train_rauc = self.my_auc_score(train_ture_y, train_pred_sy)
            train_loss = self.sess.run(self.my_loss(train_ture_y, train_pred_y))
            print('Epoch: %d\ttrain dauc: %.2f\ttrain rauc: %.2f\ttrain loss: %.2f\tval dauc: %.2f\tval rauc: %.2f\tval loss: %.2f' %
                  (0, train_dauc, train_rauc, train_loss, val_dauc, val_rauc, val_loss))
            f.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (0, train_dauc, train_rauc, train_loss, val_dauc, val_rauc, val_loss))

        for epoch in range(1, epochs+1):
            start = time.time()
            np.random.shuffle(keys)
            for k in keys:
                i = 1
                train_data = split_data.Data(data=train_x[k], label=train_y[k])
                while i > 0:
                    batch = train_data.next_batch(batch_size)
                    self.sess.run(self.step,
                                  feed_dict={self.x: batch[0],
                                             self.y_: batch[1],
                                             self.keep_prob: 0.6})
                    if i * batch_size >= train_x[k].shape[0]:
                        i = -1
                    else:
                        i += 1
            end = time.time()
            if epoch > 0:
                if val_x is not None:
                    pred_scores = []
                    ture_labels = []
                    for k in val_x.keys():
                        score = self.y_conv.eval(feed_dict={self.x: val_x[k],
                                                            self.keep_prob: 1.})
                        pred_scores.append(score)
                        ture_labels.append(val_y[k])
                    pred_y = np.vstack(pred_scores)
                    ture_y = np.vstack(ture_labels)
                    pred_sy = self.sess.run(tf.nn.softmax(pred_y))
                    val_dauc, val_rauc = self.my_auc_score(ture_y, pred_sy)
                    val_loss = self.sess.run(self.my_loss(ture_y, pred_y))
                train_pred_scores = []
                train_ture_labels = []
                for k in train_x.keys():
                    train_score = self.y_conv.eval(feed_dict={self.x: train_x[k],
                                                              self.keep_prob: 1.})
                    train_pred_scores.append(train_score)
                    train_ture_labels.append(train_y[k])
                train_pred_y = np.vstack(train_pred_scores)
                train_ture_y = np.vstack(train_ture_labels)
                train_pred_sy = self.sess.run(tf.nn.softmax(train_pred_y))
                train_dauc, train_rauc = self.my_auc_score(train_ture_y, train_pred_sy)
                train_loss = self.sess.run(self.my_loss(train_ture_y, train_pred_y))
                print('Epoch: %d\ttrain dauc: %.2f\ttrain rauc: %.2f\ttrain loss: %.2f\tval dauc: %.2f\tval rauc: %.2f\tval loss: %.2f\ttime: %.2f' %
                            (epoch, train_dauc, train_rauc, train_loss, val_dauc, val_rauc, val_loss, end-start))
                with open(self.save_dir + '/loss.txt', 'a') as f:
                    f.write('%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (epoch, train_dauc, train_rauc, train_loss, val_dauc, val_rauc, val_loss))

                if epoch % 5 == 0:
                    self.saver.save(self.sess, self.save_dir + '/tf_model', global_step=epoch)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    save_dir = 'results/MotifCNN'
    pssm_dir = 'data/'
    seq_train_dir = 'data/training/'
    seq_test_dir = 'data/test/'
    data = read_data.Input()

    epochs = 100
    batch_size = 32
    seed = 231
    with tf.Session(config=config) as sess:
        tf.set_random_seed(seed)
        x = tf.placeholder("float", shape=[None, None, 20, 1], name='x')
        y_ = tf.placeholder("float", shape=[None, 2, 2], name='y_')
        keep_prob = tf.placeholder("float", name='keep_prob')
        model = Model(sess=sess, seed=seed, save_dir=save_dir, x=x, y_=y_, keep_prob=keep_prob)

        train_x, train_y, train_names = data.get_pssm_varDic_2l(seq_train_dir + 'AAP.txt',
                                                                seq_train_dir + 'ABP.txt',
                                                                seq_train_dir + 'ACP.txt',
                                                                seq_train_dir + 'AFP.txt',
                                                                seq_train_dir + 'AHTP.txt',
                                                                seq_train_dir + 'AIP.txt',
                                                                seq_train_dir + 'AMP.txt',
                                                                seq_train_dir + 'APP.txt',
                                                                seq_train_dir + 'ATbP.txt',
                                                                seq_train_dir + 'AVP.txt',
                                                                seq_train_dir + 'CCC.txt',
                                                                seq_train_dir + 'CPP.txt',
                                                                seq_train_dir + 'DDV.txt',
                                                                seq_train_dir + 'PBP.txt',
                                                                seq_train_dir + 'QSP.txt',
                                                                seq_train_dir + 'TXP.txt',
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir,
                                                                pssm_dir)


        test_x, test_y, test_names = data.get_pssm_varDic_2l(seq_test_dir + 'AAP.txt',
                                                            seq_test_dir + 'ABP.txt',
                                                            seq_test_dir + 'ACP.txt',
                                                            seq_test_dir + 'AFP.txt',
                                                            seq_test_dir + 'AHTP.txt',
                                                            seq_test_dir + 'AIP.txt',
                                                            seq_test_dir + 'AMP.txt',
                                                            seq_test_dir + 'APP.txt',
                                                            seq_test_dir + 'ATbP.txt',
                                                            seq_test_dir + 'AVP.txt',
                                                            seq_test_dir + 'CCC.txt',
                                                            seq_test_dir + 'CPP.txt',
                                                            seq_test_dir + 'DDV.txt',
                                                            seq_test_dir + 'PBP.txt',
                                                            seq_test_dir + 'QSP.txt',
                                                            seq_test_dir + 'TXP.txt',
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir,
                                                            pssm_dir)

        print('Training Start!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        model.train(train_x, train_y, test_x, test_y, epochs, batch_size)
        print('Training End!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))




import tensorflow as tf
import numpy as np
import time
import os
import random


class denoiser(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.writer = tf.summary.FileWriter('{}/{}'.format(self.args.summary_dir, self.args.model_type))
        self.build_graph()

    def build_graph(self):
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.args.channel_num], name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.args.channel_num])

        if self.args.model_type == 'ircnn':
            self.Y = self.ircnn(self.X)
        elif self.args.model_type == 'dncnn':
            self.Y = self.dncnn(self.X)
        elif self.args.model_type == 'cnn':
            self.Y = self.cnn(self.X)
        else:
            pass
        self.loss = tf.Variable(0.0, tf.float32)
        for batch in range(self.args.batch_size):
            for channel in range(self.args.channel_num):
                self.loss = self.loss + tf.norm(
                    tf.reshape(self.Y[batch, :, :, channel], [-1]) - tf.reshape(self.Y_[batch, :, :, channel], [-1]), 2)
        self.loss = self.loss / self.args.batch_size
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def cnn(self, input, is_training=True):
        with tf.variable_scope('block1'):
            t = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
        for i in range(2, self.args.layer_num + 2):
            with tf.variable_scope('block%d' % i):
                t = tf.layers.conv2d(t, 64, 3, padding='same', name='conv%d' % i, use_bias=False)
                t = tf.nn.relu(tf.layers.batch_normalization(t, training=is_training))
        with tf.variable_scope('block%d' % (self.args.layer_num + 2)):
            t = tf.layers.conv2d(t, self.args.channel_num, 3, padding='same')
            output = t
        return output

    def dncnn(self, input, is_training=True):
        with tf.variable_scope('block1'):
            t = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
        for i in range(2, self.args.layer_num + 2):
            with tf.variable_scope('block%d' % i):
                t = tf.layers.conv2d(t, 64, 3, padding='same', name='conv%d' % i, use_bias=False)
                t = tf.nn.relu(tf.layers.batch_normalization(t, training=is_training))
        with tf.variable_scope('block%d' % (self.args.layer_num + 2)):
            t = tf.layers.conv2d(t, self.args.channel_num, 3, padding='same')
            output = t
        return input - output

    def ircnn(self, input, is_training=True):
        with tf.variable_scope('block1'):
            t = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
            temp = t

        for i in range(2, self.args.layer_num + 2, 2):
            with tf.variable_scope('block%d' % i):
                tt = tf.layers.conv2d(t, 64, 3, padding='same', name='conv%d' % i)
                tt = tf.nn.relu(tf.layers.batch_normalization(tt, training=is_training))
                tt = tf.layers.conv2d(tt, 64, 3, padding='same', name='conv%d' % (i+1))
                tt = tf.nn.relu(tf.layers.batch_normalization(tt, training=is_training))
                tt = tf.add(t, tt)
                t =  tt
        with tf.variable_scope('block%d' % (self.args.layer_num+2)):
            t = tf.layers.conv2d(t, 64, 3, padding='same', name='convm')
            t = tf.layers.batch_normalization(t, training=is_training)
            t = tf.add(t, temp, name='add')
            output = tf.layers.conv2d(t, self.args.channel_num, 1, padding='same')
            return input - output


    def make_noise(self, image):
        noised_image = image.copy()
        for num in range(image.shape[0]):
            if image.shape[3] == 1:
                for row in range(image.shape[1]):
                    choices = random.sample(range(image.shape[2]), int(self.args.percent * image.shape[2]))
                    for col in choices:
                        noised_image[num][row][col][0] = 0
            if image.shape[3] == 3:
                for row in range(image.shape[0]):
                    for channel in range(3):
                        choices = random.sample(range(image.shape[2]), int(self.args.percent * image.shape[2]))
                        for col in choices:
                            noised_image[num][row][col][channel] = 0
        return noised_image

    def save(self, iter_num):
        saver = tf.train.Saver()
        if not os.path.exists(self.args.ckpt_dir):
            os.makedirs(self.args.ckpt_dir)
        save_path = '{}/{}'.format(self.args.ckpt_dir, self.args.model_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("[*] Saving model...")
        saver.save(self.sess, os.path.join(save_path, self.args.model_type), global_step=iter_num)

    def load(self):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('{}/{}'.format(self.args.ckpt_dir, self.args.model_type))
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint('{}/{}'.format(self.args.ckpt_dir, self.args.model_type))
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def train(self, data):
        # data: num, r, g, b

        data_num = int(data.shape[0])
        train_data = data[:int(0.7*data_num)]
        valid_data = data[int(0.7*data_num):]

        iter_num = 0
        start_epoch = 0
        start_step = 0

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        lr = self.args.lr
        for epoch_num in range(start_epoch, self.args.epoch_num):
            if not (epoch_num+1) % 30:
                lr /= 10
            summary = tf.Summary(value=[tf.Summary.Value(tag='lr', simple_value=lr)])
            self.writer.add_summary(summary, epoch_num)

            np.random.shuffle(train_data)

            train_batch_num = int(train_data.shape[0] / self.args.batch_size)
            valid_batch_num = int(valid_data.shape[0] / self.args.batch_size)
            
            train_loss = 0
            for batch_id in range(start_step, train_batch_num):
                batch_image = train_data[batch_id * self.args.batch_size:(batch_id + 1) * self.args.batch_size, :, :, :]
                noise_batch_image = self.make_noise(batch_image)
                _, loss = self.sess.run([self.train_op, self.loss], 
                        feed_dict={self.Y_: batch_image, self.X: noise_batch_image, self.lr: lr, self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" % (epoch_num + 1, batch_id + 1, train_batch_num, time.time() - start_time, loss))
                train_loss += loss
                iter_num += 1
            train_loss /= train_batch_num
            summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
            self.writer.add_summary(summary, epoch_num)


            valid_loss = 0
            for batch_id in range(0, valid_batch_num):
                batch_image = train_data[batch_id * self.args.batch_size:(batch_id + 1) * self.args.batch_size, :, :, :]
                noise_batch_image = self.make_noise(batch_image)
                loss = self.sess.run([self.loss], feed_dict={self.Y_: batch_image, self.X: noise_batch_image, self.is_training: False})
                valid_loss += loss[0]

            valid_loss /= valid_batch_num
            print("Epoch: [%2d] loss: %.6f" % (epoch_num + 1, valid_loss))
            summary = tf.Summary(value=[tf.Summary.Value(tag='valid_loss', simple_value=valid_loss)])
            self.writer.add_summary(summary, epoch_num)




        self.save(iter_num)
        print("[*] Finish training.")

    def test(self, data):
        from imageio import imread, imsave
        self.sess.run(tf.global_variables_initializer())
        load_model_status, global_step = self.load()
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        output_image = self.sess.run(self.Y, feed_dict={self.X: data, self.is_training: False})
  
        if self.args.channel_num == 1:
            output_image = output_image.reshape([output_image.shape[1], output_image.shape[2]])
        elif self.args.channel_num == 3:
            output_image = output_image.reshape([output_image.shape[1], output_image.shape[2], self.args.channel_num])
        imsave('result.png', output_image)
        print("[*] Save result image.")

import tensorflow as tf
import numpy as np
import time
import os
import random

def dncnn(input, channel, layer, is_training=True):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, layer + 2):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block' + str(layer + 2)):
        output = tf.layers.conv2d(output, channel, 3, padding='same')
    return output

class denoiser(object):
    def __init__(self, sess, percent, layer, batch_size, channel_dim):
        self.sess = sess
        self.channel_dim = channel_dim
        self.percent = percent
        self.layer = layer
        self.batch_size = batch_size
        # build model

        # clean image
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.channel_dim], name='clean_image')
        # is_training or is testing
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # noised images from clean images
        self.X = tf.placeholder(tf.float32, [None, None, None, self.channel_dim])

        # noise images -> output clean images
        self.Y = dncnn(self.X, channel=self.channel_dim, layer=self.layer, is_training=self.is_training)

        self.loss = tf.Variable(0.0, tf.float32)
               
        # loss, between clean image and noised image
        for batch in range(self.batch_size):
            for channel in range(self.channel_dim):
                self.loss = self.loss + tf.norm(
                    tf.reshape(self.Y[batch, :, :, channel], [-1]) - tf.reshape(self.Y_[batch, :, :, channel], [-1]), 2)
        self.loss = self.loss / self.batch_size

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        print("[*] Initialize model successfully...")

    def make_noise(self, image):
        noised_image = image.copy()
        for num in range(image.shape[0]):
            if image.shape[3] == 1:
                for row in range(image.shape[1]):
                    choices = random.sample(range(image.shape[2]), int(self.percent * image.shape[2]))
                    for col in choices:
                        noised_image[num][row][col][0] = 0
            if image.shape[3] == 3:
                for row in range(image.shape[0]):
                    for channel in range(3):
                        choices = random.sample(range(image.shape[2]), int(self.percent * image.shape[2]))
                        for col in choices:
                            noised_image[num][row][col][channel] = 0
        return noised_image

    def save(self, iter_num, ckpt_dir, model_name='DNCNN'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def train(self, data, lr, ckpt_dir, epoch):
        # data: num, r, g, b
        numBatch = int(data.shape[0] / self.batch_size)
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        for each_epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            for batch_id in range(start_step, numBatch):
                batch_image = data[batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :, :, :]
                noised_image = self.make_noise(batch_image)
                _, loss = self.sess.run([self.train_op, self.loss],
                                        feed_dict={self.Y_: batch_image,
                                                   self.X: noised_image,
                                                   self.lr: lr[each_epoch],
                                                   self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (each_epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
        self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def test(self, data, ckpt_dir):
        from scipy.misc import imread, imsave
        tf.initialize_all_variables().run()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        output_image = self.sess.run(self.Y, feed_dict={self.X: data, self.is_training: False})
  
        if self.channel_dim == 1:
            output_image = output_image.reshape([output_image.shape[1], output_image.shape[2]])
        elif self.channel_dim == 3:
            output_image = output_image.reshape([output_image.shape[1], output_image.shape[2], self.channel_dim])
        imsave('result.png', output_image)
        print("[*] Save result image.")
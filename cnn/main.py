from model import *
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from imageio import imread, imsave
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'  

parser = argparse.ArgumentParser()

parser.add_argument('--channel_num', dest='channel_num', type=int)
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=100, help='# of epoch')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--percent', dest='percent', type=float)
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--summary_dir', dest='summary_dir', default='./summary', help='summary are saved here')
parser.add_argument('--input', dest='input', type=str)
parser.add_argument('--layer_num', dest='layer_num', type=int, default=10)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('--model_type', dest='model_type', default='ircnn', help='model type')

args = parser.parse_args()


def load_percent(image):
    pass


def load_data(channel):
    X = []
    if channel == 3:
        import pickle
        with open(r'./data/3-channel/data_batch_1', 'rb') as f:
            dict = pickle.load(f, encoding='iso-8859-1')
        data = dict['data']
        X = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        X = X[:2000]
    if channel == 1:
        import pickle
        with open(r'./data/3-channel/data_batch_1', 'rb') as f:
            dict = pickle.load(f, encoding='iso-8859-1')
        data = dict['data']
        T = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        T = T[:2000]

        T = np.dot(T[..., :3], [0.299, 0.587, 0.144]).astype(np.float32)
        for i in range(0, 2000):
            image = T[i]
            image = image.reshape([image.shape[0], image.shape[1], 1])
            X.append(image)
        X = np.array(X)
    return X


def main(_):
    if args.use_gpu:
        tf.device('/gpu:0')
    with tf.Session() as sess:
        model = denoiser(sess, args)
        if args.phase == 'train':
            model.train(load_data(args.channel_num))
        elif args.phase == 'test':
            image = np.array(imread("./data/test/" + args.input + ".png"))
            image = image.reshape([1, image.shape[0], image.shape[1], args.channel_num])
            model.test(image)
        else:
            exit(0)


if __name__ == '__main__':
    tf.app.run()

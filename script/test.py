##################################################
## Author: RuochenLiu
## Email: ruochen.liu@columbia.edu
## Version: 1.0.0
##################################################

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import cv2
import os
import os.path
from datetime import datetime
import socket
from PIL import ImageGrab, Image
import warnings
warnings.filterwarnings("ignore")

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def preprocess(im):
    # Original size is (224, 256, 3) to (70, 128, 3)
    box = (0, 60, 256, 200)
    im_cropped = im.crop(box)
    im_resize = im_cropped.resize((128, 70))
    image = np.array(im_resize)
    return image

class Replay:
    def __init__(self):
        self.buffer = []
        self.length = 0
        self.max_length = 50000

    def write(self, data):
        if self.length >= self.max_length:
            self.buffer.pop(0)
            self.length -= 1
        self.buffer.append(data)
        self.length += 1

    def read(self, batch_size):
        return random.sample(self.buffer, min(batch_size, self.length))
# Restore pre-trained weights
sess = tf.Session()
saver = tf.train.import_meta_graph('./streetfighter_model/trained_model-1200.meta')
saver.restore(sess,tf.train.latest_checkpoint('./streetfighter_model/'))
W_conv1 = sess.run('W_conv1:0')
W_conv2 = sess.run('W_conv2:0')
W_conv3 = sess.run('W_conv3:0')
b_conv1 = sess.run('b_conv1:0')
b_conv2 = sess.run('b_conv2:0')
b_conv3 = sess.run('b_conv3:0')
W_fc1 = sess.run('W_fc1:0')
b_fc1 = sess.run('b_fc1:0')
W_fc2 = sess.run('W_fc2:0')
b_fc2 = sess.run('b_fc2:0')
W_fc3 = sess.run('W_fc3:0')
b_fc3 = sess.run('b_fc3:0')
W_fc4 = sess.run('W_fc4:0')
b_fc4 = sess.run('b_fc4:0')
W_fc5 = sess.run('W_fc5:0')
b_fc5 = sess.run('b_fc5:0')

class Network:
    def __init__(self, session, image_size, ram_length, n_out):
        
        self.session = session
        self.image_size = image_size
        self.h, self.w, self.num_channels = self.image_size
        self.ram_length = ram_length
        self.n_out = n_out
        
        self.n1 = 8
        self.n2 = 16
        self.n3 = 16
        self.n4 = 512
        self.n5 = 256
        self.n6 = 128
        self.n7 = 64
         
        self.x = tf.placeholder(tf.float32, [None, self.h, self.w, self.num_channels], name='x')
        self.r = tf.placeholder(tf.float32, [None, self.ram_length], name='r')
        self.y = tf.placeholder(tf.float32, [None, self.n_out], name='y')
        self.x_image = tf.reshape(self.x, [-1, self.h, self.w, self.num_channels])
        
        self.W_conv1 = tf.get_variable('W_conv1', initializer = W_conv1)
        self.b_conv1 = tf.get_variable('b_conv1', initializer = b_conv1)
        self.h_conv1 = tf.nn.relu(tf.add(conv(self.x_image, self.W_conv1), self.b_conv1))
        
        self.W_conv2 = tf.get_variable('W_conv2', initializer = W_conv2)
        self.b_conv2 = tf.get_variable('b_conv2', initializer = b_conv2)
        self.h_conv2 = tf.nn.relu(tf.add(conv(self.h_conv1, self.W_conv2), self.b_conv2))
       
        self.W_conv3 = tf.get_variable('W_conv3', initializer = W_conv3)
        self.b_conv3 = tf.get_variable('b_conv3', initializer = b_conv3)
        self.h_conv3 = tf.nn.relu(tf.add(conv(self.h_conv2, self.W_conv3), self.b_conv3))
        
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 144*self.n3]) # Change dim
        self.W_fc1 = tf.get_variable('W_fc1', initializer = W_fc1)
        self.b_fc1 = tf.get_variable('b_fc1', initializer = b_fc1)
        self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_conv3_flat, self.W_fc1), self.b_fc1))
        
        self.W_fc2 = tf.get_variable('W_fc2', initializer = W_fc2)
        self.b_fc2 = tf.get_variable('b_fc2', initializer = b_fc2)
        self.h_fc2 = tf.nn.relu(tf.add(tf.matmul(tf.concat([self.h_fc1, self.r], 1), self.W_fc2), self.b_fc2))
        
        self.W_fc3 = tf.get_variable('W_fc3', initializer = W_fc3)
        self.b_fc3 = tf.get_variable('b_fc3', initializer = b_fc3)
        self.h_fc3 = tf.nn.relu(tf.add(tf.matmul(self.h_fc2, self.W_fc3), self.b_fc3))

        self.W_fc4 = tf.get_variable('W_fc4', initializer = W_fc4)
        self.b_fc4 = tf.get_variable('b_fc4', initializer = b_fc4)
        self.h_fc4 = tf.nn.relu(tf.add(tf.matmul(self.h_fc3, self.W_fc4), self.b_fc4))

        self.W_fc5 = tf.get_variable('W_fc5', initializer = W_fc5)
        self.b_fc5 = tf.get_variable('b_fc5', initializer = b_fc5)
        self.q = tf.add(tf.matmul(self.h_fc4, self.W_fc5), self.b_fc5, name='q')
        
        
        
        self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
    
    def compute(self, x, r):
        return self.session.run(self.q, feed_dict={self.x:np.reshape(x,[-1, self.h, self.w, self.num_channels]), self.r:np.reshape(r,[-1, self.ram_length])})
    
    def train(self, x_batch, r_batch,  y_batch):
        _ = self.session.run(self.train_step, feed_dict={self.x: x_batch, self.r: r_batch, self.y: y_batch})

class Player:
    def __init__(self, tf_session):
        self.image_size = (70, 128, 3)
        self.ram_length = 30
        self.h, self.w, self.num_channels = self.image_size
        self.n_out = 21
        self.total_reward = 0
        self.gamma = 0.9
        self.epsilon = 0.05
        self.batch_size = 120
        self.replay = Replay()
        self.q = Network(tf_session, self.image_size, self.ram_length, self.n_out)

    def gather_exp(self, last_observation, last_ram, action, reward, observation ,ram):
        self.replay.write((last_observation, last_ram, action, reward, observation, ram))

    def choose_action(self, observation, ram):
        if np.random.rand() > self.epsilon:
            q_compute = self.q.compute(observation, ram)
            return np.argmax(q_compute)
        else:
            return np.random.choice(list(range(self.n_out)))

    def q_update(self):
        sars_batch = self.replay.read(self.batch_size)

        q_last = self.q.compute([s[0] for s in sars_batch], [s[1] for s in sars_batch])
        q_this = np.zeros_like(q_last)

        index_not_none = [i for i in range(np.shape(sars_batch)[0]) if sars_batch[i][4] is not None]
        q_this_not_none = self.q.compute([sb[4] for sb in sars_batch if sb[4] is not None], [sb[5] for sb in sars_batch if sb[5] is not None])

        for i in range(len(index_not_none)):
            q_this[index_not_none[i],:] = q_this_not_none[i,:]
        
        x_batch = np.zeros([np.shape(sars_batch)[0],self.h, self.w, self.num_channels])
        r_bacth = np.zeros([np.shape(sars_batch)[0], self.ram_length])
        y_batch = np.zeros([np.shape(sars_batch)[0], self.n_out])

        for i in range(np.shape(sars_batch)[0]):
            x_batch[i,:] = sars_batch[i][0]
            r_bacth[i,:] = sars_batch[i][1]
            for j in range(self.n_out):
                if j == sars_batch[i][2]:
                    y_batch[i,j] = sars_batch[i][3] + self.gamma * np.max(q_this[i])
                else:
                    y_batch[i,j] = q_last[i,j]

        self.q.train(x_batch, r_bacth, y_batch)

    def set_epsilon(self, epsilon):
        self.epsilon = 0.01 + (1.0 - 0.01)*np.exp(-0.001*epsilon)

    def reset_epsilon(self):
        self.epsilon = 0.0

    def gather_reward(self, reward):
        self.total_reward += reward # User current HP diff as total reward

    def get_total_reward(self):
        return self.total_reward

    def set_total_reward(self, new_total):
        self.total_reward = new_total

class Console:
    def __init__(self, host = '127.0.0.1', port = 8001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.connection, self.address = self.sock.accept()
        print('Python got a client at {}'.format(self.address))

    def recv(self):
        self.buffer = self.connection.recv(1024).decode()
        return self.buffer

    def send(self, msg):
        _ = self.connection.send(msg.encode())

    def close(self):
        _ = self.connection.close()

MODEL_TEST_DIR = './streetfighter_model/test/'
if not os.path.exists(MODEL_TEST_DIR):
    os.makedirs(MODEL_TEST_DIR)
date_object = datetime.now()
current_time = date_object.strftime('%H:%M:%S')
print('Waiting for client -- {}'.format(current_time))
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = tf.Session()
ep_rewards = [] # Collect reward after each move
ep_win = []
player = Player(sess)
console = Console()
# Check the client is ready
_ = console.recv()

sess.run(tf.global_variables_initializer())      
saver = tf.train.Saver()
num_win = 0
fights = []
current_time = date_object.strftime('%H:%M:%S')
print("Testing starts -- {}".format(current_time))

for ep in range(1200):
    
    player.set_total_reward(0)
    action = np.random.choice(list(range(player.n_out)))
    console.send('{}\n'.format(action+1))
    _ = console.recv()
    observation = preprocess(ImageGrab.grabclipboard())
    console.send('0\n')
    buf = console.recv()
    feedback = buf.split(' ')
    ram = np.array([int(x) for x in feedback[:30]])
    reward = int(feedback[30])
    done = int(feedback[31])
    player.gather_reward(reward)
    last_observation = observation
    last_ram = ram
    rival = np.argmax(ram[-16:])
    for i in range(11880):
        action = player.choose_action(last_observation, last_ram)
        console.send('{}\n'.format(action+1))
        _ = console.recv()
        observation = preprocess(ImageGrab.grabclipboard())
        console.send('0\n')
        buf = console.recv()
        feedback = buf.split(' ')
        ram = np.array([int(x) for x in feedback[:30]])
        reward = int(feedback[30])
        done = int(feedback[31])

        if done != 0:
            observation = None
            ram = None

        last_observation = observation
        last_ram = ram

        if done != 0:
            ep_rewards.append(player.get_total_reward())
            ep_win.append(done)
            num_win = num_win + 1 if done == 1 else num_win
            break
            
    fights.append([rival, done, ep_rewards[-1]])
    if (ep+1) % 50 == 0:
        date_object = datetime.now()
        current_time = date_object.strftime('%H:%M:%S')
        print('After {} rounds, won {} times -- {}'.format(ep+1, num_win, current_time))

console.close()
fights = np.array(fights)
np.savetxt(MODEL_TEST_DIR + "{}_win.csv".format(ep+1), ep_win, delimiter=",")
np.savetxt(MODEL_TEST_DIR + "{}_fights.csv".format(ep+1), fights, delimiter=",")

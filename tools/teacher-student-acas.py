#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

    
def lenet(args, x, size):
    shape = args.studentargs

    output = tf.reshape(x, shape=[-1, size])
    layer_sizes = [int(arg) for arg in shape[1:]]
    for i, layer_size in enumerate(layer_sizes[:-1]):
        output = tf.layers.dense(output,
                                 layer_size,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                     scale=0.01),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(
                                     scale=0.01),
                                 activation=tf.nn.relu,
                                 name='student_dense%d_layer' % i)
    output = tf.layers.dense(output,
                             layer_sizes[-1],
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                 scale=0.01),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(
                                 scale=0.01),
                             name='student_output_layer')
    return tf.identity(output, name='student_model_output')


VERTICAL_SEPARATIONS = [0, 1, 5, 10, 20, 40, 60, 80, 100]


def parse_args():
    parser = argparse.ArgumentParser(description='teacher-student model')
    parser.add_argument('--model', dest='model', default='saved_model', help="model_path to save the student model\n In testing, give trained student model.", type=str)
    parser.add_argument('--task', dest='task', help='task for this file, train/test/val', type=str)
    parser.add_argument('--lr', dest='lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--epoch', dest='epoch', default=100000000, help='total epoch', type=int)
    parser.add_argument('--batchsize', dest='batchsize', default=1024, type=int)
    parser.add_argument('--stopping_criteria', default='No', type=str)
    parser.add_argument('--timeout', default=7200, help='distillation timeout time.', type=int)
    parser.add_argument('--threshold', default=0.9, help='distillation threshold', type=float)
    parser.add_argument('--studentargs', nargs='+', type=str, help='The arguments for the student model')
    parser.add_argument('--stop', type=str)
    return parser.parse_args()


def train(args):
    print("Task: distillition")

    acc_table = [ False for i in range(0,101)]

    data, label = load_training_data()
    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    valid_data = data[:5000]
    data = data[5000:]
    valid_label = label[:5000]
    label = label[5000:]

    print( len(data), len(label), len(valid_data), len(valid_label))

    batch_size = args.batchsize
    learning_rate = args.lr
    model_path = args.model
    total_epoch = args.epoch

    Y = tf.placeholder(tf.float32, shape = (None, 5), name='teacher_argmax')
    X = tf.placeholder(tf.float32, shape = (None, 7), name='student_input')
    student = lenet(args, X, 7)

    tf_loss = tf.nn.l2_loss(tf.stop_gradient(Y) - student)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
    gpu_options = tf.GPUOptions()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=200)

    index=np.array(range(len(data)))    # index randomly ordered
    iterations = len(data)/batch_size

    cnt=0
    begin = time.time()
    for i in xrange(total_epoch):
        np.random.shuffle(index)
        cost_sum=0
        for j in xrange(iterations):
            batch_x = np.float32(data[index[j*batch_size:(j+1)*batch_size]])
            batch_y = np.float32(label[index[j*batch_size:(j+1)*batch_size]])
            _, cost = sess.run([optimizer1, tf_loss], feed_dict={X : batch_x, Y: batch_y})
            cost_sum += cost
        cnt += 1
        avg_time = time.time()-begin

        ra = relative_accuracy(student, X, valid_data, valid_label, sess)
        print ("epoch %d - avg. %f seconds in each epoch, lr = %.0e, cost = %f , avg-cost-per-logits = %f, ra_acc = %.4f"%(i, avg_time/cnt,learning_rate, cost_sum, cost_sum/iterations, ra))

        ra_int = int(ra * 100)

        if acc_table[ra_int] == False:
            for k in range(0, ra_int+1):
                acc_table[k] = True

            saver.save(sess, os.path.join(model_path, str(ra_int)))
            total_time = time.time()-begin
            f = open(os.path.join(model_path, 'ra.txt'),'a')
            f.write('time ' + str(total_time) + " acc " + str(ra) + '\n')
            f.close()

        if avg_time > args.timeout:
            print('Distillation stopped due to timeout.')
            break

    saver.save(sess, os.path.join(model_path, 'final'))

    total_time = time.time()-begin
    f = open(os.path.join(args.model, 'log.txt'), 'w')
    f.write('time ' + str(total_time) + '\n')
    f.close()


def relative_accuracy(student, input, v_data, v_label, sess):
    
    p = sess.run(student, feed_dict={input : v_data})
    pred = np.argmin(p, axis=1)
    true = np.argmin(v_label, axis=1)

    acc = np.mean(pred == true)
    return acc

def test_student(args):
    #print "Task : test\n"
    X = tf.placeholder(tf.float32, shape = (None, 7), name='student_input')
    student = lenet(args, X, 7)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.device('/cpu:0'):
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.model, 'final'))

    batch_size = args.batchsize
    data, label = load_testing_data()
    begin = time.time()

    p = sess.run(student, feed_dict={X : data})
    pred = np.argmin(p, axis=1)
    true = np.argmin(label, axis=1)

    #print([np.sum(pred == i) for i in range(5)])

    end = time.time()

    acc = np.mean(pred == true)
    f = open(os.path.join(args.model, 'log.txt'), 'a')
    f.write("acc = %f .  Computing time = %f seconds"%(acc, end-begin) + '\n')
    f.close()


def load_caffe_teacher(nnet):
    rpx_infile = open(nnet, 'r')
    readline = lambda: rpx_infile.readline().strip()

    line = readline()

    # Ignore the comments
    while line.startswith('//'):
        line = readline()

    # Parse the dimensions
    all_dims = [int(dim) for dim in line.split(',')
                if dim != '']
    nb_layers, input_size, \
        output_size, max_lay_size = all_dims

    # Get the layers size
    line = readline()
    nodes_in_layer = [int(l_size_str) for l_size_str in line.split(',')
                           if l_size_str != '']
    assert(input_size == nodes_in_layer[0])
    assert(output_size == nodes_in_layer[-1])

    # Load the symmetric parameter
    line = readline()
    is_symmetric = int(line.split(',')[0]) != 0
    # if symmetric == 1, enforce that psi (input[2]) is positive
    # if to do so, it needs to be flipped, input[1] is also adjusted
    # In practice, all the networks released with Reluxplex 1.0 have it as 0
    # so we will just ignore it.


    # Load Min/Max/Mean/Range values of inputs
    line = readline()
    inp_mins = [float(min_str) for min_str in line.split(',')
                     if min_str != '']
    line = readline()
    inp_maxs = [float(max_str) for max_str in line.split(',')
                     if max_str != '']
    line = readline()
    inpout_means = [float(mean_str) for mean_str in line.split(',')
                         if mean_str != '']
    line = readline()
    inpout_ranges = [float(range_str) for range_str in line.split(',')
                          if range_str != '']
    assert(len(inp_mins) == len(inp_maxs))
    assert(len(inpout_means) == len(inpout_ranges))
    assert(len(inpout_means) == (len(inp_mins) + 1))

    # Load the weights
    parameters = []
    for layer_idx in range(nb_layers):
        # Gather weight matrix
        weights = []
        biases = []
        for tgt_neuron in range(nodes_in_layer[layer_idx+1]):
            line = readline()
            to_neuron_weights = [float(wgt_str) for wgt_str in line.split(',')
                                 if wgt_str != '']
            assert(len(to_neuron_weights) == nodes_in_layer[layer_idx])
            weights.append(to_neuron_weights)
        for tgt_neuron in range(nodes_in_layer[layer_idx+1]):
            line = readline()
            neuron_biases = [float(bias_str) for bias_str in line.split(',')
                             if bias_str != '']
            assert(len(neuron_biases) == 1)
            biases.append(neuron_biases[0])
        assert(len(weights) == len(biases))
        parameters.append([np.array(weights).T, np.array(biases)])

    # for i in range(input_size):
    #     print((inp_mins[i] - inpout_means[i])/inpout_ranges[i])
    #     print((inp_maxs[i] - inpout_means[i])/inpout_ranges[i])
    #     print()


    # create the tf model
    data = tf.placeholder(tf.float32, shape = (None, input_size), name='teacher_input')
    output = tf.contrib.layers.flatten(data)

    for i in range(nb_layers-1):
        output = tf.layers.dense(output,
                                 nodes_in_layer[i+1],
                                 kernel_initializer = tf.constant_initializer(parameters[i][0]),
                                 bias_initializer = tf.constant_initializer(parameters[i][1]),
                                 activation=tf.nn.relu,
                                 use_bias = True)
                                 #,name='teacher_dense%d_layer' % i)
    output = tf.layers.dense(output,
                             output_size,
                             kernel_initializer = tf.constant_initializer(parameters[-1][0]),
                             bias_initializer = tf.constant_initializer(parameters[-1][1]),
                             use_bias = True)
                             #,name='teacher_output_layer')
    output = tf.identity(output, name='teacher_model_output')
    
    return data, output


def load_torch_teacher(nnet):
    rpx_infile = open(nnet, 'r')
    readline = lambda: rpx_infile.readline().strip()

    line = readline()

    # Ignore the comments
    while line.startswith('//'):
        line = readline()

    # Parse the dimensions
    all_dims = [int(dim) for dim in line.split(',')
                if dim != '']
    nb_layers, input_size, \
        output_size, max_lay_size = all_dims

    # Get the layers size
    line = readline()
    nodes_in_layer = [int(l_size_str) for l_size_str in line.split(',')
                           if l_size_str != '']
    assert(input_size == nodes_in_layer[0])
    assert(output_size == nodes_in_layer[-1])

    # Load the symmetric parameter
    line = readline()
    is_symmetric = int(line.split(',')[0]) != 0
    # if symmetric == 1, enforce that psi (input[2]) is positive
    # if to do so, it needs to be flipped, input[1] is also adjusted
    # In practice, all the networks released with Reluxplex 1.0 have it as 0
    # so we will just ignore it.


    # Load Min/Max/Mean/Range values of inputs
    line = readline()
    inp_mins = [float(min_str) for min_str in line.split(',')
                     if min_str != '']
    line = readline()
    inp_maxs = [float(max_str) for max_str in line.split(',')
                     if max_str != '']
    line = readline()
    inpout_means = [float(mean_str) for mean_str in line.split(',')
                         if mean_str != '']
    line = readline()
    inpout_ranges = [float(range_str) for range_str in line.split(',')
                          if range_str != '']
    assert(len(inp_mins) == len(inp_maxs))
    assert(len(inpout_means) == len(inpout_ranges))
    assert(len(inpout_means) == (len(inp_mins) + 1))

    # Load the weights
    parameters = []
    for layer_idx in range(nb_layers):
        # Gather weight matrix
        weights = []
        biases = []
        for tgt_neuron in range(nodes_in_layer[layer_idx+1]):
            line = readline()
            to_neuron_weights = [float(wgt_str) for wgt_str in line.split(',')
                                 if wgt_str != '']
            assert(len(to_neuron_weights) == nodes_in_layer[layer_idx])
            weights.append(to_neuron_weights)
        for tgt_neuron in range(nodes_in_layer[layer_idx+1]):
            line = readline()
            neuron_biases = [float(bias_str) for bias_str in line.split(',')
                             if bias_str != '']
            assert(len(neuron_biases) == 1)
            biases.append(neuron_biases[0])
        assert(len(weights) == len(biases))
        parameters.append([np.array(weights).T, np.array(biases)])

    # for i in range(input_size):
    #     print((inp_mins[i] - inpout_means[i])/inpout_ranges[i])
    #     print((inp_maxs[i] - inpout_means[i])/inpout_ranges[i])
    #     print()


    # create the pytorch model
    net = TorchNet(parameters)
    
    return net


class TorchNet(nn.Module):
    def __init__(self,  parameters):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(5, 50)
        self.fc1.weight.data = torch.Tensor(parameters[0][0].T)
        self.fc1.bias.data = torch.Tensor(parameters[0][1].T)
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data = torch.Tensor(parameters[1][0].T)
        self.fc2.bias.data = torch.Tensor(parameters[1][1].T)
        self.fc3 = nn.Linear(50, 50)
        self.fc3.weight.data = torch.Tensor(parameters[2][0].T)
        self.fc3.bias.data = torch.Tensor(parameters[2][1].T)
        self.fc4 = nn.Linear(50, 50)
        self.fc4.weight.data = torch.Tensor(parameters[3][0].T)
        self.fc4.bias.data = torch.Tensor(parameters[3][1].T)
        self.fc5 = nn.Linear(50, 50)
        self.fc5.weight.data = torch.Tensor(parameters[4][0].T)
        self.fc5.bias.data = torch.Tensor(parameters[4][1].T)
        self.fc6 = nn.Linear(50, 50)
        self.fc6.weight.data = torch.Tensor(parameters[5][0].T)
        self.fc6.bias.data = torch.Tensor(parameters[5][1].T)
        self.fc7 = nn.Linear(50, 5)
        self.fc7.weight.data = torch.Tensor(parameters[6][0].T)
        self.fc7.bias.data = torch.Tensor(parameters[6][1].T)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        output = self.fc7(x)
        return output


def load_training_data():
    data = np.load('./acas/train_data.npy')
    label = np.load('./acas/train_label.npy')
    tmp = np.argmin(label, axis=1)
    #print([np.sum(tmp == i) for i in range(5)])
    return data,label


def load_testing_data():
    data = np.load('./acas/test_data.npy')
    label = np.load('./acas/test_label.npy')
    return data,label


def generate_acas_data(size):
    '''
    x_min = np.array([0.0,-3.141593,-3.141593,100.0,0.0,])
    x_max = np.array([60760.0,3.141593,3.141593,1200.0,1200.0,])
    x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0,])
    x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0,])

    x_min_norm = (x_min - x_mean)/x_range
    x_max_norm = (x_max - x_mean)/x_range
    print(x_min_norm)
    print(x_max_norm)
    '''

    data = []
    for i in range(size):
        input1 = np.random.uniform(low=-0.328422877151, high=0.679857768706)
        input2 = np.random.uniform(low=-0.500000055133, high=0.500000055133)
        input3 = np.random.uniform(low=-0.500000055133, high=0.500000055133)
        input4 = np.random.uniform(low=-0.5, high=0.5)
        input5 = np.random.uniform(low=-0.5, high=0.5)
        #input6 = np.random.randint(5)
        #input7 = VERTICAL_SEPARATIONS[np.random.randint(9)]
        #data += [[input1, input2, input3, input4, input5, input6, input7]]
        data += [[input1, input2, input3, input4, input5]]
    data = np.array(data)
    return data


def gen_data(args):

    data = generate_acas_data(1000)
    label = []

    #data_res = [[],[],[],[],[]]
    #label_res = [[],[],[],[],[]]

    data_res = []
    label_res = []

    tf_teachers = []
    for a in range(1,2):#,6):
        for b in range(1,2):#,10):
            DEFAULT_MODEL = './acas/nnet/ACASXU_run2a_'+str(a)+'_'+ str(b)+'_batch_2000.nnet'
            x = load_caffe_teacher(DEFAULT_MODEL)
            tf_teachers += [(x)]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    for a in range(1,6):
        for b in range(1,10):
            model = './acas/nnet/ACASXU_run2a_'+str(a)+'_'+ str(b)+'_batch_2000.nnet'
            
            torch_teacher = load_torch_teacher(model)
            torch_teacher.eval()
            torch.onnx.export(torch_teacher, torch.zeros(1,5), f'acas_{a}_{b}.onnx', verbose=True)
    exit()


    flag = True
    for x in data:
        '''
        a = int(x[-2])
        b = VERTICAL_SEPARATIONS.index(x[-1])
        c = 9 * a + b
        if c != 0:
            continue
        X, teacher = teachers[c]
        '''
        X, tf_teacher = tf_teachers[0]
        tf_pred = sess.run(tf_teacher, feed_dict={X : [x]})
        torch_pred = torch_teacher.forward(torch.Tensor(x))
        print(x, tf_pred, torch_pred)
        
        exit()

        data_res += [x]
        label_res +=[pred[0]]
        
        #data_res[pred_argmin] += [x]
        #label_res[pred_argmin] += [pred[0]]

        '''
        a = [len(x) for x in data_res]
        if np.min(a) == 20000:
            break
        '''

    '''
    even_data = []
    even_label = []
    for i in range(5):
        even_data += data_res[i][0:20000]
        even_label += label_res[i][0:20000]
    even_data = np.array(even_data)
    even_label = np.array(even_label)
    '''
    even_data = np.array(data_res)
    even_label = np.array(label_res)
    
    print(even_data.shape)
    print(even_label.shape)

    np.save('./acas/train_data.npy', even_data)
    np.save('./acas/train_label.npy', even_label)


if __name__ == '__main__':

    args = parse_args()

    if args.task == 'train':
        with tf.device('/gpu:0'):
            train(args)
    elif args.task == 'gen_data':
        gen_data(args)
    elif args.task == 'test_student':
        test_student(args)
    else:
        assert False

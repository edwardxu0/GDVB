import os
import numpy as np
import scipy.io as sio


class Onnx2VerifierTranslator():

    def __init__(self):
        pass

    def translate(self, onnx_model, target_verifier, output_dir, input_format, verifier_format):
        self.onnx_model = onnx_model
        self.output = os.path.join(output_dir, onnx_model.name)
        self.input_format = input_format
        self.verifier_format = verifier_format
        getattr(self, 'to_'+target_verifier)()

    
    def to_planet(self):
        lines = []
        layer_neurons_names = []
        nb_relu = 0
        nb_linear = 0
        arch = self.onnx_model.arch

        layers = enumerate(arch)
        for i,l in layers:
            if l.type =="Input":
                nb_neurons = l.in_shape.prod()
                names = ['inX'+str(x) for x in range(nb_neurons)]
                layer_neurons_names += [names]

                for i in range(len(names)):
                    lines += ['Input ' + names[i] + '\n']

            elif l.type == "FC":
                if i == len(arch) - 1:
                    temp_pre = 'Linear outX'
                elif arch[i+1].type != 'ReLU':
                    temp_pre = 'Linear lina'
                    nb_linear += 1
                elif arch[i+1].type == 'ReLU':
                    next(layers)
                    temp_pre = 'ReLU relu:'+str(nb_relu)+':'
                    nb_relu += 1

                #print(l.weights.shape[0],l.weights.shape[1],len(layer_neurons_names[-1]))

                neuron_names = []
                for i in range(l.weights.shape[0]):
                    temp = temp_pre + str(i) + ' ' + str(l.bias[i])
                    neuron_names += [temp.split(' ')[1]]
                    for j in range(l.weights.shape[1]):
                        temp += ' '+ str(l.weights[i][j])+ ' ' + layer_neurons_names[-1][j]
                    lines += [temp + '\n']

                layer_neurons_names += [neuron_names]

            elif l.type == ["Flatten"]:
                print('Ignoring layer type: ' + l.type)
                continue

            else:
                print('Unsupported layer type: ' + l.type)

        open(self.output + '.rlv', 'w').writelines(lines)
 

    def to_reluplex(self):
        self.to_planet()


    def to_mipverify(self):
        weights_bias_dict = {}
        mip_layers = []
        lines = []
        lines += ['using MIPVerify, Gurobi, MAT']#, Images, Colors']
        lines += ['pd = matread("{}")'.format(self.output+'.mat')]

        for i,l in enumerate(self.onnx_model.arch):
            if l.type =="Input":
                continue
            elif l.type == "FC":
                W = l.weights
                B = l.bias
                if W.shape[1] != B.shape[0]:
                    W = W.T
                assert W.shape[1] == B.shape[0]
                #print(W.shape, B.shape)
                weights_bias_dict['l'+str(i)+'/weight'] = W
                weights_bias_dict['l'+str(i)+'/bias'] = B
                lines += ['l{}_w = pd["l{}/weight"]'.format(i,i)]
                lines += ['l{}_b = pd["l{}/bias"]'.format(i,i)]
                if str(l.bias.shape) == "(1,)":
                    lines += ['l{} = Linear(l{}_w, [l{}_b])'.format(i,i,i)]
                else:
                    lines += ['l{} = Linear(l{}_w, squeeze(l{}_b, 1))'.format(i,i,i)]
                mip_layers += ['l'+str(i)]

            elif l.type == "Conv":
                W = l.weights
                B = l.bias
                #print(W.shape)
                if self.input_format != self.verifier_format:
                    #print("Before",W.shape,B.shape)
                    W = W.transpose(2,3,1,0)
                    #print("After",W.shape,B.shape)
                    print("Transposing the conv weights shape to NHWC format for MIPVerify.")
                #print(W.shape, B.shape)
                weights_bias_dict['l'+str(i)+'/weight'] = W
                weights_bias_dict['l'+str(i)+'/bias'] = B
                lines += ['l{}_w = pd["l{}/weight"]'.format(i,i)]
                lines += ['l{}_b = pd["l{}/bias"]'.format(i,i)]
                lines += ['l{} = Conv2d(l{}_w, squeeze(l{}_b, 1))'.format(i,i,i)]
                mip_layers += ['l'+str(i)]

            elif l.type == "ReLU":
                mip_layers += ['ReLU()']
            elif l.type == "Flatten":
                #TODO figure out the number here
                mip_layers += ['Flatten({})'.format(4)]
            elif l.type == "Pool":
                if l.pool_type == 'max':
                    pool_type = 'MIPVerify.maximum'
                elif l.pool_type == 'average':
                    pool_type = 'Base.mean'
                else:
                    assert False
                lines +=['l{} = Pool((1,{},{},1), {})'.format(i,l.stride,l.stride, pool_type)]
                mip_layers += ['l'+str(i)]
            else:
                assert False, 'Unsupported layer type:' + l.type
        lines += ['layers = [' + (','.join(mip_layers))+ ']']
        #lines += ['nn = Sequential(layers, "{}.n2")'.format(self.onnx_model.name)]
        #lines += []
        lines = [l+'\n' for l in lines]

        open(self.output+'.jl', 'w').writelines(lines)
        sio.savemat(self.output+'.mat', weights_bias_dict)


    def to_neurify(self):
        supported_layers = ['Input', 'FC', 'Conv']

        nnet_nb_layers = -1
        nnet_layer_types = []
        nnet_layer_sizes = []
        conv_layers_info = []
        layer_input_shape = []

        weights_bias = []

        for l in self.onnx_model.arch:
            if l.type =="Input":
                input_shape = l.in_shape
                layer_input_shape += [input_shape]
                nnet_layer_sizes += [np.prod(input_shape)]

            elif l.type == "FC":
                nnet_layer_types += [0]
                nnet_layer_sizes += [l.size]
                weights_bias += [(l.weights, l.bias)]
                #print("FC",l.size,l.weights.shape)

            elif l.type == "Conv":
                nnet_layer_types += [1]

                padding = l.padding

                width = layer_input_shape[-1][1]
                kernel_size = l.kernel_size
                stride = l.stride
                output_w = (((width-kernel_size+2*padding) // stride)+1)
                size = l.size * output_w * output_w

                if len(conv_layers_info) == 0:
                    input_channel = 3
                else:
                    input_channel = conv_layers_info[-1][0]
                layer_input_shape += [[l.size, output_w, output_w]]
                conv_layers_info += [[l.size, input_channel, kernel_size, stride, padding]]
                nnet_layer_sizes += [size]
                weights_bias += [(l.weights, l.bias)]

                #print(size, '==', np.prod(l.out_shape))
                #print([l.size, output_w, output_w],"==", l.out_shape)
                #print([l.size, input_channel, kernel_size, stride, padding])
                #print(l.weights.shape)
                #print(l.bias.shape)
                #print()
            else:
                continue
            nnet_nb_layers += 1


        # process layer info
        lines = []
        #lines = ['//'+','.join(map(str,[l.type for l in arch]))]

        lines += ["{},{},{},{},".format(nnet_nb_layers, nnet_layer_sizes[0], nnet_layer_sizes[-1], max(nnet_layer_sizes))]
        lines += [','.join(map(str, nnet_layer_sizes))+',']
        lines += [','.join(map(str, nnet_layer_types))+',']

        for c in conv_layers_info:
            lines += [','.join(map(str, c))+',']

        # process weight and bias
        for i in range(len(nnet_layer_types)):
            w = weights_bias[i][0]
            b = weights_bias[i][1]

            # TODO: for some network?
            #transpose the weights of the fully connected layer
            #if nnet_layer_types[i] == 0:
            #    w = w.T

            for x in w:
                #flatten the weights for convlayer
                if nnet_layer_types[i] == 1:
                    x = x.transpose(0,2,1)
                    x = x.flatten()
                lines += [",".join(map(str,x))+',']
            for x in b:
                lines += [str(x)+',']

        lines = '\n'.join(map(str,lines))

        open(self.output + '.nnet', 'w').writelines(lines)

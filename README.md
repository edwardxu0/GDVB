# GDVB(Generative Diverse DNN Verification Benchmarks)

## Directory Structure
* configs: configuration files of GDVB.
* data: training data.
* gdvb: source code.
* lib: dependencies
* results: results will be stored here.
* tmp: temporary folder.
* tools: tools to generate properties and so on.


## Overview
The use of GDVB includes two phases. The first phase includes generation of the benchmark, and the second phase contains running the verification tools. After these steps, we can collect and analyze the results. In our paper, the study was conducted on a server with NVIDIA 1080Ti GPUs for training and 2.3GHz or 2.2GHz Xeon processors for verification. The two artifacts(MNIST-conv_big and DAVE-2) took about 426.1 and 930.6 hours to run respectively. To demonstrate the usage of the GDVB tool in a reasonable amount of resource and time, we designed a toy MNIST example(MNIST_tiny). In the following, we will describe the factor configuration of MNIST_tiny, how to use the GDVB to generate the DNN verification benchmark and run verification with several tools, and finally we will analyze and discuss the results.

GDVB starts from an seed verification problem. In this case, it's the robustness check on the MNIST_tiny network. The MNIST_tiny network has 3 hidden layers with 50 neurons in each. The detailed GDVB configurations are stored in `configs/mnist_tiny.toml`. We used six factors, # of neurons, # of FC layers, input domain size, input dimensions, property scale, and property translate. We used pairwise coverage strength, and 2 levels for all factors except for the # of neuron, which has 4 levels. In the configuration file, we can also define hyper-parameters such as, the strength of the covering array, the number of training epochs, the epsilon value of the local robustness property, the time and memory limit for verification, and which verification tools to run, etc. The above settings generates an MCA of size 8. The detailed MCA will be stored at `results/mnist_tiny.[seed]/ca.txt`. After running the GDVB benchmark generation and verification, the logs and results will be saved in the `results` folder.


## Usage
1. Load the virtual environment and get help.
```
source openenv.sh
python -m gdvb -h
```

2. Generate the GDVB Benchmark.

   + Generate the Mix Covering Array.
   ```
   python -m gdvb configs/mnist_tiny.toml gen_ca 10
   ```
   + Train the neural networks.
   ```
   python -m gdvb configs/mnist_tiny.toml train 10
   ```
   + Generate properties.
   ```
   python -m gdvb configs/mnist_tiny.toml gen_props 10
   ```

3. Run verification tools with the generated benchmark.
```
python -m gdvb configs/mnist_tiny.toml verify 10
```

4. Collect and analyze results.
```
python -m gdvb configs/mnist_tiny.toml analyze 10
```


## Results
TODO ... make tables, plots,

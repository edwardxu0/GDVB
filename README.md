# Systematic Generation of Diverse Benchmarks for DNN Verification(GDVB) V1.0.3

  * [Overview](#overview)
  * [Inputs](#inputs)
  * [Installation](#installation)
  * [Usage - Toy Example](#usage---toy-example)
  * [Usage - Full Study](#usage---full-study)
  * [Description of Source Code](#description-of-source-code)
  * [Disclaimer](#disclaimer)
  * [Acknowledgements](#acknowledgements)

## Overview
GDVB allows for the generation of diverse DNN verification benchmarks that is described in the this paper, [Systematic Generation of Diverse Benchmarks for DNN Verification](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_5).

While focus of GDVB is on benchmark generation, to illustrate its utility in this artifact we also include support for running a set of verifiers on the generated benchmarks.

This documentation explains the usage of the included GDVB implementation including: the commands, the inputs, and expected output.  The GDVB paper includes two large case studies applied to non-trivial DNNs.  While generating the benchmark configurations in the studies takes little time, training the test cases and running the verifiers on the benchmarks took several hundreds of hours on powerful GPU servers.  Consequently, to demonstrate the end-to-end usage of GDVB tool in a reasonable amount time, we designed a toy MNIST example(MNIST_tiny), on which this document mainly focuses.  Note that this toy example is not included in the paper.

## Inputs
GDVB is built based on [R4V](https://arxiv.org/abs/1908.08026) and [DNNV](https://github.com/dlshriver/DNNV). The inputs to GDVB is a configuration file, the task to perform, and a random seed, and the details can be viewed in the tool's help prompt. The configuration file contains the factor level descriptions for the MCA problem. In addition, we can also define hyper-parameters such as, the strength of the covering array, the number of training epochs, the epsilon value of the local robustness property, the time and memory limit for verification, and which verification tools to run, etc.

The factors that the current implementation supports are: Number of Neurons, FC Layers, Conv Layers, Input Dimensions, Input Size, Property Scale, and Property Translation. In the configuration file, we can define a set of factors with a closed interval where both borders are inclusive and the number of levels for each factor respectively, e.g. for the number of neurons, if the levels are `neu=3` and the interval is `neu=[0.5,1.0]`, this means in the resulting benchmark, we will be choosing from the number of neurons selection from (0.5, 0.75, 1.0) times the number of neurons in the seed neural network.

In addition to DNN benchmark generation, we also provided an easy way to run a set of verification tools which can be set in the configuration file as well. The verification tools that we currently support are: Reluplex, Plant, Bab, BaBSB, Neurify, ERAN_DZ, ERAN_DP, ERAN_RZ, and ERAN_RZ.

## Installation
1. Acquire the [ACTS](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software) covering array generation tool. Store the `jar` file as `lib/acts.jar`.

2. Install dependencies.(Use the following command for Ubuntu based OSes.)
+ (Required) Dependencies for GDVB.
```
sudo apt install python3.6-venv python3.6-dev cmake protobuf-compiler default-jre
```
+ (Optional) Dependencies for Planet verification tool.
```
sudo apt install libglpk-dev libltdl-dev qt5-default valgrind
```
3. Create a virtual environment
```
.env.d/manage.sh init
```
4. Load the virtual environment
```
. .env.d/openenv.sh
```
5. Install R4V and DNNV
```
.env.d/manage.sh install R4V DNNV
```
6. (Optional) Get a [Gurobi Lisense](https://www.gurobi.com/) and store it at `lib/gurobi.lic`. This step is necessary if you plan to use the eran_refinezono, eran_refinepoly, bab, or babsb verification tools.

## Usage - Toy Example
In this section, we focus on the MNIST_tiny as an example, and how to use the GDVB to generate the DNN verification benchmark and run verification with several tools, and finally we will analyze the results and depict useful information.

GDVB starts from a seed verification problem. In this case, it involves verifying a local robustness property on the MNIST_tiny network. The MNIST_tiny network has 3 hidden layers with 50 neurons in each. For this problem, the configuration file is stored as `configs/mnist_tiny.toml`. In the configuration file, we used six factors, # of neurons, # of FC layers, input domain size, input dimensions, property scale, and property translate. We chose pairwise coverage strength, and 2 levels for all factors except for the # of fully connected layers, which has 4 levels.

1. The GDVB uses a command-line interface, firstly we open a terminal, load the virtual environment and show tool's help.
```
. .env.d/openenv.sh
python -m gdvb -h
```

2. Generate the GDVB Benchmark.

   + Generate the Mix Covering Array.
   ```
   python -m gdvb configs/mnist_tiny.toml gen_ca 10
   ```
   This step will generate the MCA at `./results/mnist_tiny.10/ca.txt`. This file contains the factor-level description of the GDVB benchmark, refer to the ACTS documentation for more details. Below are the first few lines of the generated covering array. For the computed test cases, the parameters are the verifier performance factors and their assignments are chosen the levels which are 0 indexed.
    ```
    # Degree of interaction coverage: 2
    # Number of parameters: 6
    # Maximum number of values per parameter: 4
    # Number of configurations: 8
    ------------Test Cases--------------

    Configuration #1:

    1 = neu=0
    2 = fc=0
    3 = idm=1
    4 = ids=1
    5 = eps=1
    6 = prop=1
    ...
    ```

   + Train the neural networks.
   ```
   python -m gdvb configs/mnist_tiny.toml train 10
   ```
   This step will create the R4V configurations and distill the 8 neural networks as defined by the MCA. The resulting neural networks will be saved at `./results/mnist_tiny.10/dis_model/`.

   + Generate properties.
   ```
   python -m gdvb configs/mnist_tiny.toml gen_props 10
   ```
   This step will create the transformed properties as defined by the MCA. They will be saved at `./results/mnist_tiny.10/props`.

3. Run verification tools with the generated benchmark.
```
python -m gdvb configs/mnist_tiny.toml verify 10
```
This step runs the verification tools(eran_deepzono,eran_deeppoly, neurify, planet, and reluplex) over the benchmark. We limited the selection of verifiers to those that do not require the use of any licensed libraries for convenience in running the artifact.  The results are saved at `./results/mnist_tiny.10/veri_log`.

4. Collect and analyze results.
```
python -m gdvb configs/mnist_tiny.toml analyze 10
```
This step collects the results from the verification tools and generates tables. The verification results, depicted in the form of SCR and PAR-2 scores, are shown here:
|       Verifier|              SCR|            PAR-2|
|---------------|-----------------|-----------------|
|  eran_deepzono|                7|       20.9464625|
|  eran_deeppoly|                7|         15.93765|
|        neurify|                8|        0.7614125|
|         planet|                7|        18.753625|
|       reluplex|                8|        1.0453875|

Note that the results you get may differ slightly, e.g., due to stochasticity in training the neural networks and hardware of your machine.  However, the trend between verification tools should be similar. 

5. The above commands for "MNIST_tiny" can also be run in the combined script `./scripts/run_mnist_tiny.sh`. The example results can be found in the `example_results` folder and a sample execution log is stored at `example_/mnist_tiny.log.txt` file.

## Usage - Full Study
!!! WARNING: The full study of two artifacts(MNIST-conv_big and DAVE-2) took 426.1 and 930.6 hours to run respectively with NVIDIA 1080Ti GPUs for training and 2.3/2.2GHz Xeon processors for verification. We do not recommend running it without extensive resource.

This following script will run the full study test cases to be used to generate results that are described in the paper.
```
./scripts/run_study.sh
```

## Description of Source Code

### Directory Structure
* configs: configuration files of GDVB.
* data: training data.
* gdvb: source code.
* lib: dependencies.
* results: results will be stored here.
* tmp: temporary folder.
* tools: tools to generate properties and so on.

The "gdvb" directory contains the source code of the implementation in addition to the "tools" folder. The "tools" folder contains the code to generate the properties to the verification problems. Except to the helper code, there are four main components in the "gdvb" directory. The "nn" component contains the methods to parse neural network in the "onnx" format and some utility functions. The "network.py" contains the methods to configure, train, and verify a neural network with the connection of "R4V" and "DNNV". The "genbench.py" contains the main functions to configure the MCA from GDVB configuration file, generating the MCA, creating the GDVB benchmark code, training, verifying, and depicting results, etc.

## Disclaimer
1. [R4V](https://arxiv.org/abs/1908.08026) is still a private project that is under development. We included a snapshot of R4V in `lib/R4V` as it is a crucial part of GDVB. Please consider not to use R4V for other purposes than GDVB until it is officially released to the public. At that point, we will also update GDVB to accommodate the changes in the latest R4V.

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation under grant numbers 1901769 and 1900676, by the U.S. Army Research Office under grant number W911NF-19-1-0054.

# Systematic Generation of Diverse Benchmarks for DNN Verifiers(GDVB) V1.0.3

- [GDVB (Generation of Diverse DNN Verification problem Benchmarks)](#gdvb-generative-diverse-dnn-verification-benchmarks-)
  * [Overview](#overview)
  * [Supported Factors and Verification Tools](#supported-factors-and-verification-tools)
  * [Description of MCA](#description-of-mca)
  * [Usage - Toy Example](#usage---toy-example)
  * [Usage - Full Study](#usage---full-study)
  * [Description of Source Code](#description-of-source-code)
    + [Directory Structure](#directory-structure)
  * [Disclaimer](#disclaimer)

## Overview
GDVB allows for the generation of diverse DNN verification benchmarks.  While its focus is on benchmark generation, to illustrate its utility in this artifact we also include support for running a set of verifiers on the generated benchmarks.

This documentation explains the usage of the included GDVB implementation including: the commands, the inputs, and expected output.  The CAV paper includes two large case studies applied to non-trivial DNNs.  While generating the benchmark configurations in the studies takes little time, training the test cases and running the verifiers on the benchmarks took several hundreds of hours on powerful GPU servers.  Consequently, to demonstrate the end-to-end usage of GDVB tool in a reasonable amount time, we designed a toy MNIST example(MNIST_tiny), on which this decoment mainly focuses.  Note that this toy example is not included in the CAV paper.

## Supported Factors and Verification Tools
Modularity was an important consideration when implementing GDVB. The implementation is designed to support features to be added in future. The factors that the current implementation supports are: Number of Neurons, FC Layers, Conv Layers, Input Dimensions, Input Size, Property Scale, and Property Translation. The verification tools that we currently support are shown in the following table.
| Verifier |  Algorithmic Approach |
|----------|-----------------------|
| Reluplex | Search-Optimization   |
| Plant    | Search-Optimization   |
| Bab      | Search-Optimization   |
| BaBSB    | Search-Optimization   |
| Neurify  | Optimization          |
| ERAN_DZ  | Reachability          |
| ERAN_DP  | Reachability          |
| ERAN_RZ  | Reachability          |
| ERAN_RZ  | Reachability          |

## Installation
### Dependencies
This project is built based on [R4V](https://arxiv.org/abs/1908.08026) and [DNNV](https://github.com/dlshriver/DNNV).

1. Acquire the [ACTS](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software) covering array generation tool. Store the `jar` file as `lib/acts.jar`.

2. Install dependencies.(Use the following command for Ubuntu based OSes.)
```
sudo apt install default-jre cmake qt4-qmake
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
6. (Optional) Get a [Gurobi Lisense](https://www.gurobi.com/) and store it at `lib/gurobi.lic`. This step is required if you plan to use the eran_refinezono, eran_refinepoly, bab, or babsb verification tools.

## Description of MCA


## Usage - Toy Example
In this section, we focus on the MNIST_tiny as an example, and how to use the GDVB to generate the DNN verification benchmark and run verification with several tools, and finally we will analyze the results and depict useful information.

GDVB starts from a seed verification problem. In this case, it involves verifying a local robustness property on the MNIST_tiny network. The MNIST_tiny network has 3 hidden layers with 50 neurons in each. The inputs to the GDVB tools is a configuration file, the task to perform, and a random seed, and the details can be viewed in the tool's help prompt. For this problem, the configuration file is stored as `configs/mnist_tiny.toml`. In the configuration file, we used six factors, # of neurons, # of FC layers, input domain size, input dimensions, property scale, and property translate. We chose pairwise coverage strength, and 2 levels for all factors except for the # of fully connected layers, which has 4 levels. In addition, we can also define hyper-parameters such as, the strength of the covering array, the number of training epochs, the epsilon value of the local robustness property, the time and memory limit for verification, and which verification tools to run, etc. The above settings generates an MCA of size 8. The detailed MCA will be stored at `results/mnist_tiny.[seed]/ca.txt`. After running the GDVB benchmark generation and verification, the logs and results will be saved in the `results` folder. We will discuss the commands and their detailed explanation as below.

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
   This step will generate the MCA at `./results/mnist_tiny.10/ca.txt`. This file contains the factor-level description of the GDVB benchmark, refer to the ACTS documentation for more details(see below in Disclaimer 1). This step took less than 1 second on our platform. Below are the first few lines of the generated covering array. For the computed test cases, the parameters are the verifier performance factors and their assignments are chosen the levels which are 0 indexed.
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
   This step will create the R4V configurations and distill the 8 neural networks as defined by the MCA. The resulting neural networks will be saved at `./results/mnist_tiny.10/dis_model/`. This step took around 14 minutes on our platform.

   + Generate properties.
   ```
   python -m gdvb configs/mnist_tiny.toml gen_props 10
   ```
   This step will create the transformed properties as defined by the MCA. They will be saved at `./results/mnist_tiny.10/props`. This step took around 4 seconds on our platform.

3. Run verification tools with the generated benchmark.
```
python -m gdvb configs/mnist_tiny.toml verify 10
```
This step runs the verification tools(eran_deepzono,eran_deeppoly, neurify, planet, and reluplex) over the benchmark. We limited the selection of verifiers to those that do not require the use of any licensed libraries for convenience in running the artifact.  The results are saved at `./results/mnist_tiny.10/veri_log`. This step took less than 3 minutes on our platform.

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

5. The above commands for "MNIST_tiny" can also be run in the combined script `./scripts/run_mnist_tiny.sh`. The example results can be found in the `../example_results` folder and a sample execution log is stored at `../mnist_tiny.log.txt` file.

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
# [The **GDVB** Framework] Systematic Generation of Diverse Benchmarks for DNN Verification

## I. Overview
The GDVB framework, which was developed for the paper titled "[Systematic Generation of Diverse Benchmarks for DNN Verification](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_5)," serves as a means to showcase the functionality of the proposed methods outlined in the paper and to reproduce the evaluation results presented within. This artifact encompasses several components, including 1) guidelines for installing the GDVB framework and its dependencies, 2) a tutorial on generating a small DNN verification benchmark using a toy MNIST seed problem, 3) detailed information about the configurable parameters, and 4) instructions and scripts to fully replicate the research results. It is important to note that the GDVB tool is implemented in the Python programming language, thus users are expected to possess a certain level of familiarity with Python, as well as neural network training and verification techniques.

GDVB supports generating NNV benchmarks using Fully Connected and Convolutional seed neural networks. From GDVB version 2.0.0 on, users can scale up seed verification problems. This means the benchmark instances can have more neurons, more FC and Conv layers, and more input dimensions than the original seed network. Refer to section IV for more information.


## II. Installation
1. Acquire the [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) environment management tool. Install and activate the `gdvb` conda environment.

   ```shell
   conda env create --name gdvb -f .env.d/env.yml
   . .env.d/openenv.sh
   ```

2. Acquire the [ACTS](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software) covering array generation tool and store the `jar` file as `lib/acts.jar`.

3. Install dependencies.

   3.1 (Required) Get [R4V](https://github.com/edwardxu0/r4v) and install it's conda environment.

     ```shell
     git clone https://github.com/edwardxu0/R4V.git lib/R4V
     conda env create --name r4v -f ${R4V}/.env.d/env.yml
     ```

   3.2 (Optional) Get [SwarmHost](https://github.com/edwardxu0/SwarmHost/) and install it's conda environment. Follow it's guide to install the tool and verifiers. Make sure to install SwarmHost in a separate conda environment named `swarmhost`.

     ```shell
     git clone https://github.com/edwardxu0/SwarmHost.git lib/SwarmHost
     conda env create --name swarmhost -f ${SwarmHost}/.env.d/env.yml
     ```

   3.3 (Optional) Get [DNNV](https://github.com/dlshriver/dnnv) if you wish to use the verifiers in DNNV. Follow its instructions to install the tool and verifiers. Make sure to install DNNV in a separate conda environment named `dnnv`.

   3.4 (Optional) Get [DNNF](https://github.com/dlshriver/dnnf) if you wish to use the falsifiers in DNNF. Follow its instructions to install the tool. Make sure to install DNNF in a separate conda environment named `dnnf`.

4. The environment variables are stored in the `.env.d/openenv.sh` file. Modifying the values within this file can potentially resolve certain dependency issues, such as the location of GUROBI licenses.

## III. Tutorial on GDVB
In this section, we employ GDVB on the toy "MNIST_tiny" seed verification problem to produce a small DNN verification benchmark and execute the verification process within the SwarmHost framework using the abcrown verifier. Subsequently, we analyze verification results with the SCR and PAR2 metrics.

The "MNIST_tiny" seed verification problem includes a small neural network and a local robustness property. The neural network has 3 hidden layers with 50 neurons in each. The local robustness property is defined on epsilon of 0.02. The configuration file is stored as `configs/mnist_tiny.toml`. It defines two factors, the number of neurons and number of FC layers, where each has three evenly distributed levels, e.g., 1/3, 2/3, and 1 scaling ratio compared to the seed network.

1. The GDVB framework employs a command-line interface for its operations. To ensure the correct installation of GDVB, one should load the conda environment and consult the help manual.
```
. .env.d/openenv.sh
gdvb -h

   __________ _    ______ 
  / ____/ __ \ |  / / __ )
 / / __/ / / / | / / __  |
/ /_/ / /_/ /| |/ / /_/ / 
\____/_____/ |___/_____/  
                          
usage: GDVB [-h] [--seed SEED] [--result_dir RESULT_DIR] [--override] [--debug] [--dumb]
            configs {C,T,P,V,A,E}

Generative Diverse Verification Benchmarks for Nueral Network Verification

positional arguments:
  configs               Configurations file.
  {C,T,P,V,A,E}         Select tasks to perform, including, compute [C]A, [T]rtain benchmark
                        networks, generate [P]roperties, [V]erify benchmark instances, [A]nalyze
                        benchmark results, and [E]verything above.

options:
  -h, --help            show this help message and exit
  --seed SEED           Random seed.
  --result_dir RESULT_DIR
                        Result directory.
  --override            Override existing logs.
  --debug               Print debug log.
  --dumb                Silent mode.
```

2. Generate the verification Benchmark.

   2.1) Generate the Mix Covering Array.
   ```
   gdvb configs/mnist_tiny.toml C
   
      __________ _    ______ 
     / ____/ __ \ |  / / __ )
    / / __/ / / / | / / __  |
   / /_/ / /_/ /| |/ / /_/ / 
   \____/_____/ |___/_____/  
                             
   [INFO] 02/14/2024 12:19:12 PM : Computing Factors 
   [INFO] 02/14/2024 12:19:12 PM : Computing Covering Array 
   [INFO] 02/14/2024 12:19:12 PM : # problems: 9 
   [INFO] 02/14/2024 12:19:12 PM : Computing DNN Specifications 
   [INFO] 02/14/2024 12:19:12 PM : # NN: 9 
   [INFO] 02/14/2024 12:19:12 PM : Spent 0.240134 seconds. 
   ```

   This step will generate the MCA at `./results/mnist_tiny.0/cas/ca_*.txt`. This file contains the factor-level description of the benchmark instances. Below are the first few lines of the generated covering array. For each test case, the parameters are the verification performance factors and their assignments are chosen the levels, which are 0 indexed. For more detailed information, please consult the ACTS documentation. 
    ```
   # Degree of interaction coverage: 2
   # Number of parameters: 2
   # Maximum number of values per parameter: 3
   # Number of configurations: 9
   ------------Test Cases--------------

   Configuration #1:

   1 = neu=0
   2 = fc=0

   -------------------------------------

   Configuration #2:

   1 = neu=0
   2 = fc=1
    ```

   2.2) Train the neural networks.
   ```
   gdvb configs/mnist_tiny.toml T

      __________ _    ______ 
     / ____/ __ \ |  / / __ )
    / / __/ / / / | / / __  |
   / /_/ / /_/ /| |/ / /_/ / 
   \____/_____/ |___/_____/  
                             
   [INFO] 02/14/2024 01:30:37 PM : Computing Factors 
   [INFO] 02/14/2024 01:30:37 PM : Computing Covering Array 
   [INFO] 02/14/2024 01:30:37 PM : # problems: 9 
   [INFO] 02/14/2024 01:30:37 PM : Computing DNN Specifications 
   [INFO] 02/14/2024 01:30:37 PM : # NN: 9 
   [INFO] 02/14/2024 01:30:37 PM : Training ... 
   Training ... :   0%|                                             | 0/9 [00:00<?, ?it/s]
   [INFO] 02/14/2024 01:30:37 PM : Training network: neu=0.3333_fc=0.3333_idm=1.0000_ids=1.0000_SF=0.6667 ... 
   Training ... :  11%|█████                                        | 1/9 [03:05<24:45, 185.68s/it]
   [INFO] 02/14/2024 01:33:43 PM : Training network: neu=0.3333_fc=0.6667_idm=1.0000_ids=1.0000_SF=0.6667 ... 
   Training ... :  22%|██████████                                   | 2/9 [06:26<22:43, 194.76s/it]
   [INFO] 02/14/2024 01:37:04 PM : Training network: neu=0.3333_fc=1.0000_idm=1.0000_ids=1.0000_SF=0.3333 ... 
   Training ... :  33%|███████████████▎                             | 3/9 [09:45<19:39, 196.56s/it]
   [INFO] 02/14/2024 01:40:23 PM : Training network: neu=0.6667_fc=0.3333_idm=1.0000_ids=1.0000_SF=1.3333 ... 
   Training ... :  44%|████████████████████▍                        | 4/9 [13:09<16:37, 199.47s/it]
   [INFO] 02/14/2024 01:43:47 PM : Training network: neu=0.6667_fc=0.6667_idm=1.0000_ids=1.0000_SF=1.3333 ... 
   Training ... :  56%|█████████████████████████                    | 5/9 [16:17<13:01, 195.42s/it]
   [INFO] 02/14/2024 01:46:55 PM : Training network: neu=0.6667_fc=1.0000_idm=1.0000_ids=1.0000_SF=0.6667 ... 
   Training ... :  67%|██████████████████████████████               | 6/9 [19:43<09:56, 198.81s/it]
   [INFO] 02/14/2024 01:50:20 PM : Training network: neu=1.0000_fc=0.3333_idm=1.0000_ids=1.0000_SF=2.0000 ... 
   Training ... :  78%|███████████████████████████████████          | 7/9 [22:54<06:32, 196.39s/it]
   [INFO] 02/14/2024 01:53:32 PM : Training network: neu=1.0000_fc=0.6667_idm=1.0000_ids=1.0000_SF=2.0000 ... 
   Training ... :  89%|████████████████████████████████████████     | 8/9 [26:03<03:13, 193.94s/it]
   [INFO] 02/14/2024 01:56:41 PM : Training network: neu=1.0000_fc=1.0000_idm=1.0000_ids=1.0000_SF=1.0000 ... 
   Training ... : 100%|█████████████████████████████████████████████| 9/9 [29:28<00:00, 196.50s/it]
   [INFO] 02/14/2024 02:00:06 PM : Spent 1768.792314 seconds. 
   
   ```

   The R4V configurations, as specified by the MCA, will be generated in this stage and the neural networks will be distilled accordingly. The distilled neural network models in ONNX format will be stored in the directory `results/mnist_tiny.0/dis_model/`, while the training logs will be saved in `results/mnist_tiny.0/dis_log/`.

   2.3 (Optional) Generate properties. 
   ```
   configs/mnist_tiny.toml P
   
      __________ _    ______ 
     / ____/ __ \ |  / / __ )
    / / __/ / / / | / / __  |
   / /_/ / /_/ /| |/ / /_/ / 
   \____/_____/ |___/_____/  
                           
   [INFO] 02/14/2024 11:16:31 PM : Computing Factors 
   [INFO] 02/14/2024 11:16:31 PM : Computing Covering Array 
   [INFO] 02/14/2024 11:16:31 PM : # problems: 9 
   [INFO] 02/14/2024 11:16:31 PM : Computing DNN Specifications 
   [INFO] 02/14/2024 11:16:31 PM : # NN: 9 
   [INFO] 02/14/2024 11:16:31 PM : Generating properties ... 
   Generating ... : 100%|█████████████████████████████████████████████| 9/9 [00:01<00:00,  5.81it/s]
   [INFO] 02/14/2024 11:16:33 PM : Spent 2.003998 seconds.
   ```
   This step generates the transformed properties defined by the MCA. These properties will then be saved at the location `results/mnist_tiny.0/props`. Note that this step is unnecessary and will be subsumed by the verification step, if one plan to use the verification pipeline in GDVB, e.g., SwarmHost, DNNV, and DNNF.

3. Run verification tools with the generated benchmark.
   ```
   gdvb configs/mnist_tiny.toml V
   
      __________ _    ______ 
     / ____/ __ \ |  / / __ )
    / / __/ / / / | / / __  |
   / /_/ / /_/ /| |/ / /_/ / 
   \____/_____/ |___/_____/  
                             
   [INFO] 02/14/2024 11:57:14 PM : Computing Factors 
   [INFO] 02/14/2024 11:57:14 PM : Computing Covering Array 
   [INFO] 02/14/2024 11:57:14 PM : # problems: 9 
   [INFO] 02/14/2024 11:57:14 PM : Computing DNN Specifications 
   [INFO] 02/14/2024 11:57:14 PM : # NN: 9 
   [INFO] 02/14/2024 11:57:14 PM : Verifying ... 
   Verifying ... :   0%|                                             | 0/9 [00:00<?, ?it/s]
   [INFO] 02/14/2024 11:57:14 PM : Verifying neu=0.3333_fc=0.3333_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=0.6667 with SwarmHost:[abcrown] ... 
   Verifying ... :  11%|█████                                        | 1/9 [00:12<01:37, 12.18s/it]
   [INFO] 02/14/2024 11:57:27 PM : Verifying neu=0.3333_fc=0.6667_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=0.6667 with SwarmHost:[abcrown] ... 
   Verifying ... :  22%|██████████                                   | 2/9 [00:18<01:01,  8.80s/it]
   [INFO] 02/14/2024 11:57:33 PM : Verifying neu=0.3333_fc=1.0000_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=0.3333 with SwarmHost:[abcrown] ... 
   Verifying ... :  33%|███████████████                              | 3/9 [00:24<00:44,  7.42s/it]
   [INFO] 02/14/2024 11:57:39 PM : Verifying neu=0.6667_fc=0.3333_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=1.3333 with SwarmHost:[abcrown] ... 
   Verifying ... :  44%|████████████████████                         | 4/9 [00:30<00:33,  6.73s/it]
   [INFO] 02/14/2024 11:57:45 PM : Verifying neu=0.6667_fc=0.6667_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=1.3333 with SwarmHost:[abcrown] ... 
   Verifying ... :  56%|█████████████████████████                    | 5/9 [00:35<00:24,  6.19s/it]
   [INFO] 02/14/2024 11:57:50 PM : Verifying neu=0.6667_fc=1.0000_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=0.6667 with SwarmHost:[abcrown] ... 
   Verifying ... :  67%|██████████████████████████████               | 6/9 [00:40<00:17,  5.84s/it]
   [INFO] 02/14/2024 11:57:55 PM : Verifying neu=1.0000_fc=0.3333_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=2.0000 with SwarmHost:[abcrown] ... 
   Verifying ... :  78%|███████████████████████████████████          | 7/9 [00:48<00:12,  6.46s/it]
   [INFO] 02/14/2024 11:58:03 PM : Verifying neu=1.0000_fc=0.6667_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=2.0000 with SwarmHost:[abcrown] ... 
   Verifying ... :  89%|████████████████████████████████████████     | 8/9 [00:53<00:06,  6.16s/it]
   [INFO] 02/14/2024 11:58:08 PM : Verifying neu=1.0000_fc=1.0000_idm=1.0000_ids=1.0000_eps=1.0000_prop=0_SF=1.0000 with SwarmHost:[abcrown] ... 
   Verifying ... : 100%|█████████████████████████████████████████████| 9/9 [00:59<00:00,  6.58s/it]
   [INFO] 02/14/2024 11:58:14 PM : Spent 59.608237 seconds. 
   ```
   This step runs the SwarmHost verification framework with the abcrown verifier over the benchmark as defined in configuration file of "MNIST_tiny". The verification results are saved in `results/mnist_tiny.0/veri_log`.

4. Collect and analyze results.
```
gdvb configs/mnist_tiny.toml A
   __________ _    ______ 
  / ____/ __ \ |  / / __ )
 / / __/ / / / | / / __  |
/ /_/ / /_/ /| |/ / /_/ / 
\____/_____/ |___/_____/  
                          
[INFO] 02/15/2024 02:17:39 PM : Computing Factors 
[INFO] 02/15/2024 02:17:39 PM : Computing Covering Array 
[INFO] 02/15/2024 02:17:40 PM : # problems: 9 
[INFO] 02/15/2024 02:17:40 PM : Computing DNN Specifications 
[INFO] 02/15/2024 02:17:40 PM : # NN: 9 

|       Verifier |             SCR |           PAR-2|
|----------------|-----------------|----------------|
|        abcrown |               9 |            1.75|
[INFO] 02/15/2024 02:17:40 PM : Spent 0.33223 seconds. 
```
The results obtained from the verification tools are gathered in this step, wherein the SCR and PAR-2 metrics are computed. It should be noted that the outcomes obtained may exhibit slight variations, for instance, the stochasticity nature of neural networks training and the hardware specifications used for verification. Nevertheless, the overall pattern observed among verification tools should remain consistent.

## IV. GDVB Configuration File
This section describes the parameters used in the configuration file of GDVB. There are four main parts, including configs to DNN, covering array, training, and verification parameters. This implementation of GDVB supports executing training and verification tasks on both local machines or clusters using SLURM server. Below describes the complete set of parameters for each section.
1. Deep neural network configs.
   ```toml
   [dnn]
       # dataset
      artifact = 'MNIST'
      # path to seed network in ONNX format
      onnx = './configs/networks/mnist_conv_big.onnx'
      # path to base R4V configs
      r4v_config = './configs/networks/mnist_conv_big.toml'
   ```
2. Covering Array configs.
   ```toml
   [ca]
      # covering array strength
      strength = 2

      # verification performance factors
      [ca.parameters]
         # number of levels of each factor
         [ca.parameters.level]
            neu = 5
            fc = 3
            conv = 5
            idm = 5	
            ids = 5
            eps = 5
            prop = 5

         # evenly distributed level range
         # GDVB v2: Set upper bound of range to be above 1 to scale up the seed verification problem
         [ca.parameters.range]
            neu = ['1/5','1']
            fc = ['0','1']
            conv = ['0','1']
            idm = ['1/5','1']
            ids = ['1/5','1']
            eps = ['1/5','1']
            prop = ['0','4']

      # covering array constraints
      [ca.constraints]
         # prevent invalid network specifications
         value = ['fc=0 => conv!=0',
         '(fc=2 && conv=0 && idm=0) => neu<4']
   ```
3. Training configs.
   ```toml
   [train]
      # number of epochs
      epochs = 10
      # strategies to drop a layer, choose from ['random'], more to be developed
      drop_scheme = 'random'

      # ways to dispatch training tasks
      [train.dispatch]
         # platform to use, choose from ['local', 'slurm']
         platform = 'local'
         # number of CPUs to use(only works with slurm platform)
         nb_cpus = 8
         # number of GPUs to use(only works with slurm platform)
         nb_gpus = 1
   ```
4. Verification configs.
   ```toml
   [verify]
      # epsilon radius of robustness property
      eps = 0.02
      # time limit
      time = 14400
      # memory limit
      memory = '64G'

      # choice of Verification Platform and Verifiers
      [verify.verifiers]
         # requires full installation of DNNV, including all verifiers
         DNNV = ['eran_deepzono',
               'eran_deeppoly',
               'eran_refinezono',
               'eran_refinepoly',
               'neurify',
               'planet',
               'bab',
               'bab_sb',
               'reluplex',
               'marabou',
               'nnenum',
               'mipverify',
               'verinet']
         
         # requires installation of DNNF
         DNNF = ['default']

         # requires installation of of SwarmHost
         SwarmHost = ['abcrown',
                  'neuralsat',
                  'mnbab',
                  'nnenum',
                  'verinet']

      # ways to dispatch verification tasks
      [verify.dispatch]
         # platform to use, choose from ['local', 'slurm']
         platform = "local"
         # number of CPUs to use(only works with slurm platform)
         nb_cpus = 8
         # number of GPUs to use(only works with slurm platform)
         nb_gpus = 1
   ```
## V. Result Replication
It should be noted that the comprehensive study outlined in the paper, which includes two artifacts (MNIST-conv_big and DAVE-2), required 426.1 and 930.6 hours of execution time respectively. This was accomplished using NVIDIA 1080Ti GPUs for training and 2.3/2.2GHz Xeon processors for verification. It is strongly advised against attempting to run this study without substantial resources at hand. This following script will run the full study as described in the paper.
```
./scripts/run_study.sh
```

## VI. Extending GDVB with New Features
This section provides an overview of advanced applications of GDVB for users who wish to enhance its capabilities. Various methods can be employed to achieve this, including the introduction of novel seed network architectures, properties, verifiers, and factors. In the majority of instances, users will not be required to make modifications to the source code. For example, they can incorporate new network architectures into the system by utilizing existing datasets like MNIST, CIFAR, and DAVE2, or adjust the levels and factors of existing artifacts. However, in certain cases, users may need to engage in a limited amount of coding, such as when adding a new dataset. Alternatively, more extensive coding may be necessary if there are additional functional expansions to be made to GDVB. Here is a list of four potential extensions that researchers may be interested in:
   1. *New seed network*: Obtain a seed network in the ONNX format. If a new dataset is required, suggested coding locations are `gdvb.artifacts` and `r4v.distillation.data.py`, etc.
   2. *New factors*: The fundamental components of GDVB are the factors. To introduce a new factor, it is necessary to make modifications to the core source code of GDVB, which can be found in `gdvb.core.verification_problem.py` and `gdvb.core.verification_benchmark.py`.
   3. *New seed property*: In the case of introducing a new form of verification property and desiring its integration into GDVB, it is advisable to first define a factor that will be utilized in the generation of the covering array. The modification of `gdvb.core.verification_problem.py` is required for the covering array and property generation.
   4. *New verifiers*: While GDVB provides a pipeline for executing verification tasks, the responsibility of adding new verifiers does not lie with GDVB itself. One can extend the verification frameworks, such as SwarmHost and DNNV, or employ their own scripts for verification tasks.

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation under grant numbers 1901769 and 1900676, by the U.S. Army Research Office under grant number W911NF-19-1-0054.

We greatly appreciate your enthusiasm for GDVB. Please feel free to reach out to us for any assistance or guidance regarding the utilization or expansion of GDVB.
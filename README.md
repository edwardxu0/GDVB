# Systematic Generation of Diverse Benchmarks for DNN Verifiers(GDVB) V1.0.2

## Directory Structure
* configs: configuration files of GDVB.
* data: training data.
* gdvb: source code.
* lib: dependencies
* results: results will be stored here.
* tmp: temporary folder.
* tools: tools to generate properties and so on.

## How to Use
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
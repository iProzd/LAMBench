# LAMBench [Under Construction]

LAMBench is a benchmarking tool designed to evaluate the performance of various machine learning interatomic potential models (MLIPs). It provides a comprehensive suite of tests and metrics to help developers and researchers understand the generalizability of their machine learning models.

## Features

- **Comprehensive Benchmarks**: Includes a wide range of benchmarks for different downstream tasks.
- **Easy to Use**: Simple setup and configuration to get started quickly.
- **Extensible**: Easily add new benchmarks and metrics.
- **Detailed Reports**: Generates detailed performance reports and visualizations.

## Installation

To install LAMBench, clone the repository and install the required dependencies:

```bash
pip install git+https://github.com/deepmodeling/LAMBench.git#egg=lambench
```

## Usage

To run the benchmarks, use the following command:

```bash
lambench
```

If there are errors importing the `torch` package regarding symbol error, try:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.1/site-packages/torch/lib/../../nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Contributing

We welcome contributions from the community. To contribute, please fork the repository, create a new branch, and submit a pull request with your changes.

## License

LAMBench is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

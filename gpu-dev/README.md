## Developing LightAutoML on GPU

To develop LightAutoML on GPUs using RAPIDS some prerequisites need to be met:
1. NVIDIA GPU: Pascal or higher
2. CUDA 11.0 (drivers v450.51+) or CUDA 11.2 (drivers v460.32+) need to be installed
3. Python version 3.8
4. OS: Ubuntu 16.04/18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+

### Installation

[Anaconda](https://www.anaconda.com/products/individual#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is necessary to install RAPIDS and work with environments.

1. Once you install Anaconda/Miniconda, you need to set your own environment. For example:
```bash
conda create -n Pyth38 python=3.8
conda activate Pyth38
```

2. To clone the project to your own local machine:
```bash
git clone https://github.com/Rishat-skoltech/LightAutoML_GPU.git
cd LightAutoML
```

3. Install LightAutoML in develop mode:
```bash
./build_package.sh
source ./lama_venv/bin/activate
poetry install
```
4. After that there is a `lama_venv` virtual environment in your system. To install RAPIDS you need to set up the `lama_venv` as a conda environment as well.
```bash
conda create -p ./lama_venv
```

5. To install RAPIDS for Python 3.8 and CUDA 11.0 use the following command:
```bash
conda install -p ./lama_venv -c rapidsai -c nvidia -c conda-forge rapids-blazing=21.06 python=3.8 cudatoolkit=11.0
```
You can try to install RAPIDS using other version of Python and CUDA. `rapids-blazing 21.06` is the latest version (at the time of writing this text) of BlazingSQL - a high-performance distributed SQL engine in Python, which we plan to test for some transformers. Check the [RAPIDS](https://rapids.ai/start.html) start page for the currently latest version. 

Once the RAPIDS is installed the environment is fully ready. You can activate it using the `source` command and test and implement your own code.

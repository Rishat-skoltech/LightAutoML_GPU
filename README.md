## Developing LightAutoML on GPU

To develop LightAutoML on GPUs using RAPIDS some prerequisites need to be met:
1. NVIDIA GPU: Pascal or higher
2. CUDA 11.0 (drivers v460.32+) need to be installed
3. Python version 3.8
4. OS: Ubuntu 16.04/18.04/20.04 or CentOS 7/8 with gcc/++ 9.0+

### Installation

[Anaconda](https://www.anaconda.com/products/individual#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is necessary to install RAPIDS and work with environments.

1. Once you install Anaconda/Miniconda, you need to set your own environment. For example:
```bash
conda create -n lama_venv python=3.8
conda activate lama_venv
```

2. To clone the project to your own local machine:
```bash
git clone https://github.com/Rishat-skoltech/LightAutoML_GPU.git
cd LightAutoML_GPU
```

3. Install LightAutoML in develop mode:
```bash
pip install .
pip install catboost==1.0.4
```


5. To install RAPIDS for Python 3.8 and CUDA 11.0 use the following command:
```bash
conda install -c rapidsai -c nvidia -c conda-forge rapids=22.02 cudatoolkit=11.0
pip install dask-ml
```

After you change the library code, you need to re-install the library (```pip uninstall lightautoml```). You need to go to LightAutoML directory and then install it once again, calling ```pip install .```

Please note, if you use NVIDIA GPU Ampere architecture (i.e. Tesla A100 or RTX3000 series), you may need to uninstall pytorch and install it manually 
due to compatibility issues. To do so, run following commands:
```bash
pip uninstall torch torchvision
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

Once the RAPIDS is installed, the environment is fully ready. You can activate it using the `source` command and test and implement your own code.


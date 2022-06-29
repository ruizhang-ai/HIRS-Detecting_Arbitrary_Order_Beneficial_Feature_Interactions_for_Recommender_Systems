# Instructions for reproducing the experiments 

## Environmental Requirement

0. You need to set up the environment for running the experiments (Anaconda 3-2020.02 or above and Python 3.7 or above). First, you can install Anaconda by following the instruction from  the [website](https://docs.anaconda.com/anaconda/install/).
   
   Next, you can create a virtual environment using the following commands:
   <pre><code>$ conda create -n GMCF python=3.7 anaconda
   $ conda activate GMCF</code></pre>

1. Install Pytorch with version 1.6.0 or later.

   For example (with CPU only version):
   ```
   $ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Install ```torch-geometric``` package with version 1.4.3 or later.

   *(Note that it may need to appropriatly install the package ```torch-geometric``` based on the CUDA version (or CPU version if GPU is not avaliable). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/1.4.3/notes/installation.html for more information of installing prerequisites of ```torch-geometric```)*

   For example (CUDA=10.1):
   ```
   $ CUDA=cu101
   $ TORCH=1.6.0
   $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-geometric==1.4.3
   ```
3. Download the code and data.

   clone this repository:
   ```
   $ git clone https://github.com/ruizhang-ai/HIRS_Hypergraph_Infomax_Recommender_System.git
   ```

   and go to the folder of the repository:
   ```
   $ cd HIRS_Hypergraph_Infomax_Recommender_System
   ```

   Now, our source codes are in the folder ```code/``` and the datasets are in the folder ```data/```.

4. Install other packages listed in requirements.txt.
   ```
   $ pip install -r requirements.txt
   ```

## Run the code


Go to the ```code/``` folder and run the ```main.py``` file:
   ```
   $ cd code
   $ python main.py --dataset=ml-1m --dim=64 --batch_size=512 
   ```
   For more argument options, please refer to ```main.py```


Due to the storage space limitation, we only provide the preprocessed files for the MovieLens 1M dataset and the Book-Crossing dataset. The meta file of the MovieLens 25M dataset can be found at https://grouplens.org/datasets/movielens/25m/ and we will release the methods for downloading the preprocessed files soon.



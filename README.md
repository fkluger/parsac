# PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus

If you use this code, please cite our paper:
```
@inproceedings{kluger2024parsac,
  title={PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus},
  author={Kluger, Florian and Rosenhahn, Bodo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

Related repositories:
* [HOPE-F dataset](https://github.com/fkluger/hope-f) 
* [SMH dataset](https://github.com/fkluger/smh) 
* [NYU-VP dataset](https://github.com/fkluger/nyu_vp)
* [YUD+ dataset](https://github.com/fkluger/yud_plus)
* [CONSAC](https://github.com/fkluger/consac)
* [Our J-/T-Linkage implementation for VP detection](https://github.com/fkluger/vp-linkage)

## Installation
Get the code:
```
git clone --recurse-submodules https://github.com/fkluger/parsac.git
cd parsac
git submodule update --init --recursive
```

Set up the Python environment using [Anaconda](https://www.anaconda.com/): 
```
conda env create -f environment.yml
source activate parsac
```

## Datasets

### HOPE-F
Download the [HOPE-F dataset](https://github.com/fkluger/hope-f) and extract it inside the `datasets/hope` directory.
The small dataset w/o images is sufficient for training and evaluation. 

### Synthetic Metropolis Homographies
Download the [SMH dataset](https://github.com/fkluger/smh) and extract it inside the `datasets/smh` directory.
The small dataset w/o images is sufficient for training and evaluation. 

### NYU-VP
The vanishing point labels and pre-extracted line segments for the 
[NYU dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) are fetched automatically via the *nyu_vp* 
submodule. 

### YUD and YUD+
Pre-extracted line segments and VP labels are fetched automatically via the *yud_plus* submodule. RGB images and camera 
calibration parameters, however, are not included. Download the original York Urban Dataset from the 
[Elder Laboratory's website](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) and 
store it under the ```datasets/yud_plus/data``` subfolder. 


### Adelaide-H/-F

We provide a mirror of the Adelaide dataset here: https://cloud.tnt.uni-hannover.de/index.php/s/egE6y9KRMxcLg6T.
Download it and place the `.mat` files inside the `datasets/adelaide` directory.

## Evaluation

In order to reproduce the results from the paper using our pre-trained network, 
first [download the neural network weights](https://cloud.tnt.uni-hannover.de/index.php/s/KMwborYZsbYAsd2)
and then follow the instructions on the [EVAL page](EVAL.md).

## Training

If you want to train PARSAC from scratch, please follow the instructions on the [TRAIN page](TRAIN.md).
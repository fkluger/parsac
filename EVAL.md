The `parsac.py` script can perform both training and evaluation. 
The `--eval` flag sets some default options to run evaluation on the test set. 

After following the instructions for downloading the datasets and pre-trained network weights ([README](README.md)), 
you can execute the commands below in order to reproduce the results from our paper. 

# Main Results
## Vanishing Points

### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```

### YUD
```
python parsac.py --eval --dataset yud --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```

### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --load weights/main_results/vp_nyu --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```


### YUD+
```
python parsac.py --eval --dataset yudplus --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```

## Fundamental Matrices
### HOPE-F
```
python parsac.py --eval --load ./weights/main_results/fundamental --dataset hope --data_path ./datasets/hope --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

### Adelaide
```
python parsac.py --eval --load ./weights/main_results/fundamental --dataset adelaide --data_path ./datasets/adelaide --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

## Homographies
### SMH
```
python parsac.py --eval --load weights/main_results/homography --dataset smh --data_path datasets/smh --problem homography --inlier_threshold 1e-6 --assignment_threshold 4e-6 --instances 24 --hypotheses 512 
```

### Adelaide
```
python parsac.py --eval --load weights/main_results/homography --dataset adelaide --data_path datasets/adelaide --problem homography --inlier_threshold 1e-4 --assignment_threshold 4e-3 --instances 24 --hypotheses 512 
```

# Self-Supervised Learning

##  Weighted Loss

### SU3

```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --load weights/self_supervised/weighted_su3
```

### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --load weights/self_supervised/weighted_nyu
```

### HOPE-F 
```
python parsac.py --eval --load ./weights/self_supervised/weighted_fundamental --dataset hope --data_path ./datasets/hope --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

### Adelaide-F
```
python parsac.py --eval --load ./weights/self_supervised/weighted_fundamental --dataset adelaide --data_path ./datasets/adelaide --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

##  Unweighted Loss

### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --load weights/self_supervised/unweighted_su3
```

### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --load weights/self_supervised/unweighted_nyu
```

### HOPE-F 
```
python parsac.py --eval --load ./weights/self_supervised/unweighted_fundamental --dataset hope --data_path ./datasets/hope --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

### Adelaide-F
```
python parsac.py --eval --load ./weights/self_supervised/unweighted_fundamental --dataset adelaide --data_path ./datasets/adelaide --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128 
```

# Ablation Study: Number of Model Instances
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp  --inlier_threshold 0.0001  --hypotheses 32  --load weights/ablation_instances/M_HAT --instances M_HAT
```
Replace `M_HAT` with the number of putative model instances. Valid values are {2, 3, 4, 6, 10, 12, 16}.

# Ablation Study: Weighted Inlier Counting
## with weighted inlier counting
See [main results](#main-results)

## w/o weighted inlier counting

### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/ablation_unweighted/su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --inlier_counting unweighted
```

### HOPE-F
```
python parsac.py --eval --dataset hope --data_path ./datasets/hope --problem fundamental --load ./weights/main_results/fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128  --inlier_counting unweighted
```

### SMH
```
python parsac.py --eval --load weights/main_results/homography --dataset smh --data_path datasets/smh --problem homography --inlier_threshold 1e-6 --assignment_threshold 4e-6 --instances 24 --hypotheses 512  --inlier_counting unweighted
```

# Ablation Study: Robustness to Noise and Outliers
Note: the following options only work for SU3 and Adelaide.

In order to add Gaussian noise with standard deviation `sigma` to the input observations, use the following parameter:
```
--ablation_noise sigma
```

In order to remove all ground truth outliers from the observations and then add synthetic outliers with an outlier rate of `outlier_rate`, use the following parameter:
```
--ablation_outlier_ratio outlier_rate
```


# Ablation Study: Feature Generalisation

We provide line segments extracted with [DeepLSD](https://github.com/cvg/DeepLSD) for SU3, NYU-VP and YUD(+).
Download and extract the following archive before you can run the ablation study experiments below:\
https://cloud.tnt.uni-hannover.de/index.php/s/M7TTyqGzbnCfiJX

## Train: LSD / Test: LSD
See [main results](#main-results)

## Train: LSD / Test: DeepLSD
### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/su3
```

### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --load weights/main_results/vp_nyu --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/nyu
```
### YUD
```
python parsac.py --eval --dataset yud --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/yud
```
### YUD+
```
python parsac.py --eval --dataset yudplus --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/yud
```

## Train: DeepLSD / Test: LSD
### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/ablation_features/su3_deeplsd/ --inlier_threshold 0.0001 --instances 8 --hypotheses 32  
```
### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --load weights/ablation_features/nyu_deeplsd/ --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```
### YUD
```
python parsac.py --eval --dataset yud --data_path datasets/yud_plus/data --problem vp --load weights/ablation_features/su3_deeplsd --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```
### YUD+
```
python parsac.py --eval --dataset yudplus --data_path datasets/yud_plus/data --problem vp --load weights/ablation_features/su3_deeplsd --inlier_threshold 0.0001 --instances 8 --hypotheses 32 
```

## Train: DeepLSD / Test: DeepLSD
### SU3
```
python parsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/ablation_features/su3_deeplsd/ --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/su3
```
### NYU-VP
```
python parsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --load weights/ablation_features/nyu_deeplsd/ --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/nyu
```
### YUD
```
python parsac.py --eval --dataset yud --data_path datasets/yud_plus/data --problem vp --load weights/ablation_features/su3_deeplsd --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/yud
```
### YUD+
```
python parsac.py --eval --dataset yudplus --data_path datasets/yud_plus/data --problem vp --load weights/ablation_features/su3_deeplsd --inlier_threshold 0.0001 --instances 8 --hypotheses 32  --ablation_deeplsd_folder deeplsd_features/yud
```

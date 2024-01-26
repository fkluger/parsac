Use the commands below to train PARSAC according to the experiments in our paper.

We use [Weights & Biases](https://wandb.ai/) to log the training progress. You can enable it with `--wandb online` to use online syncing or `--wandb offline` to save the logs in a folder offline.
Use the options `--wandb_entity`, `--wandb_group` and `--wandb_dir` to set the entity, group and local directory for your logs.

# Main Results
## Vanishing Points

### SU3
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine 
```

### NYU-VP
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset nyuvp --problem vp --instances 8 --hypsamples 64 --data_path datasets/nyu_vp/data --checkpoint_dir ./tmp/checkpoints --no_refine 
```


## Fundamental Matrices
### HOPE-F
```
python parsac.py --hypotheses 32 --batch 32 --samplecount 16 --inlier_threshold 0.004 --assignment_threshold 0.02 --dataset hope --problem fundamental --instances 4 --hypsamples 128 --epochs 3000 --lr_steps 2500 --data_path datasets/hope --checkpoint_dir ./tmp/checkpoints
```


## Homographies
### SMH
```
python parsac.py --hypotheses 32 --batch 4 --samplecount 8 --inlier_threshold 1e-6 --assignment_threshold 4e-6 --dataset smh --problem homography --instances 24 --hypsamples 64 --epochs 500 --lr_steps 350 --data_path datasets/smh --checkpoint_dir ./tmp/checkpoints
```

# Self-Supervised Learning

##  Weighted Loss

### SU3
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine --self_supervised
```

### NYU-VP
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset nyuvp --problem vp --instances 8 --hypsamples 64 --data_path datasets/nyu_vp/data --checkpoint_dir ./tmp/checkpoints --no_refine --self_supervised
```

### HOPE-F 
```
python parsac.py --hypotheses 32 --batch 32 --samplecount 16 --inlier_threshold 0.004 --assignment_threshold 0.02 --dataset hope --problem fundamental --instances 4 --hypsamples 128 --epochs 3000 --lr_steps 2500 --data_path datasets/hope --checkpoint_dir ./tmp/checkpoints --self_supervised
```


##  Unweighted Loss

### SU3
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine --self_supervised --cumulative_loss -1
```

### NYU-VP
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset nyuvp --problem vp --instances 8 --hypsamples 64 --data_path datasets/nyu_vp/data --checkpoint_dir ./tmp/checkpoints --no_refine --self_supervised --cumulative_loss -1
```


### HOPE-F 
```
python parsac.py --hypotheses 32 --batch 32 --samplecount 16 --inlier_threshold 0.004 --assignment_threshold 0.02 --dataset hope --problem fundamental --instances 4 --hypsamples 128 --epochs 3000 --lr_steps 2500 --data_path datasets/hope --checkpoint_dir ./tmp/checkpoints --self_supervised --cumulative_loss -1
```


# Ablation Study: Number of Model Instances
Replace `M` with the desired number of putative model instances:
```
python parsac.py --instances M --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp  --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine 
```


# Ablation Study: (Un-)weighted Inlier Counting

## w/o weighted inlier counting

### SU3
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine --inlier_counting unweighted
```

### HOPE-F
```
python parsac.py --hypotheses 32 --batch 32 --samplecount 16 --inlier_threshold 0.004 --assignment_threshold 0.02 --dataset hope --problem fundamental --instances 4 --hypsamples 128 --epochs 3000 --lr_steps 2500 --data_path datasets/hope --checkpoint_dir ./tmp/checkpoints --inlier_counting unweighted
```

### SMH
```
python parsac.py --hypotheses 32 --batch 4 --samplecount 8 --inlier_threshold 1e-6 --assignment_threshold 4e-6 --dataset smh --problem homography --instances 24 --hypsamples 64 --epochs 500 --lr_steps 350 --data_path datasets/smh --checkpoint_dir ./tmp/checkpoints --inlier_counting unweighted
```


# Ablation Study: Feature Generalisation

## Train on DeepLSD

### SU3
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine --ablation_deeplsd_folder deeplsd_features/su3
```

### NYU-VP
```
python parsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset nyuvp --problem vp --instances 8 --hypsamples 64 --data_path datasets/nyu_vp/data --checkpoint_dir ./tmp/checkpoints --no_refine --ablation_deeplsd_folder deeplsd_features/nyu
```



# TLQS Deep Learning S/W
This repository contains the S/W framework used for Deep Learning-based image super-resolution experiments in TLQS research work.
The corresponding paper will be opened and linked here once accepted.

## Software Requirements
 * tensorflow2 (<= 2.15.1)
 * python3
 * matplotlib
 * tqdm

## Instructions
### Training
 1. Set hyper-parameters properly in `main.py` such as batch size and learning rate.
 2. Run training as follows.
```
python3 main.py
```
### Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
The super-resolution performance is measured as PSNR db.
```
Epoch: 0  learning_rate: 0.0010000000474974513  Bicubic_loss: 0.002334445716184646  train loss: 0.002353070449214034
psnr_bicubic_mean: 31.771656417846682  psnr_output_mean: 32.056907653808594  diff: 0.28525123596191193
```

## Results
The experimental results will be available in the paper once it is published.

## Supported Training Algorithm
 * Mini-batch Stochastic Gradient Descent (SGD)

## Datasets
 * VDSR images / Set5 ([link](https://cv.snu.ac.kr/research/VDSR/))
 * Random sequences for shuffling data are included as 'npy' file in the repository.
 
## Questions / Comments
 * Jihyun Lim (jhades625@naver.com)
 * Sunwoo Lee (sunwool@inha.ac.kr)

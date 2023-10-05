# Task 1
[Controllable Text to Image Generation](https://proceedings.neurips.cc/paper_files/paper/2019/file/1d72310edc006dadf2190caad5802983-Paper.pdf)

## Installation of Code Requirements
1. Install Conda first. Make sure its the compatible version. Latest Cuda version isn't compatible with PyTorch. Instructions to be followed given on website. Preferred: Cuda 11.6.
2. Install Pytorch. Find the right version and follow instruction on the pytorch website. Old versions can be found in archive. Preferred: Pytorch 1.7.
3. Ensure your Cuda and Pytroch work. Following can be ran:
``` python
import torch
torch.cuda.is_available()
```

## Steps to run the code
NOTE: The code is run for BIRD dataset only as COCO model is quite big to run and re-arrange on small systems.
1. Download and place this metadata for [BIRD](https://drive.google.com/file/d/1MIpa-zWbvoY8e8YhvT4rYBNE6S_gkQMJ/view) into <mark>data</mark> folder.
2. Also in the <mark>data/birds</mark>, extract the downloaded the [Bird](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset.
3. First we have pretrain DAMSM model. This includes text encoder and image encoder. Copy the following and run in terminal. Ensure you are in the right directory.
```python
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
```
4. Train ControlGAN model for Bird Dataset.
```
python main.py --cfg cfg/train_bird.yml --gpu 0
```
To change the number of epochs, edit train_bird.yml in cfg folder. You can BATCH_SIZE and MAX_EPOCH in this file.
### Sample input/output after training
1. Input: Go to data-> birds->mytestsentence.txt. Add your input text to this file. For testing, we added: 'This bird is red and yellow.'
2. In terminal, run the following code.
```
python model_output.py --cfg cfg/train_bird_copy.yml --gpu 0
```
Output on model trained for 10 epochs
![alt text](0_s_0_g2_20epoch.png "10th epoch")
Output on pretrained model available on official Github repo.
![alt text](0_s_0_g2.png "Completely trained")

### Link to Training Dataset
[ Download Bird](https://www.vision.caltech.edu/datasets/cub_200_2011/)

### Model Parameters
Use the following code snippet to output model parameters.
For GAN, number of parameters: 102229264
```
sum(p.numel() for p in model.parameters())
```

import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm.auto import tqdm
from numba import cuda

from Basic import num_to_groups
from Model import Unet
from Trainer import Trainer

import kagglehub

# Download latest version
path = kagglehub.dataset_download("crawford/cat-dataset")

print("Path to dataset files:", path)

# load dataset from the hub

dataset = load_dataset(path="./cat_face/cat_face")
model_name = "./cat_face_model_state_dict_1000, 180.pt"
image_size = 128
channels = 3
batch_size = 16
timesteps = 1000      #총 time step
epochs = 30
training_continue = 1      #training 이어서 할지
training_state = 0         #training 단계면 1 sampling 단계면 0
check = 1                  #sampling 처음이면 1, 아니면 0
dt = 25                    #sampling 몇스텝 할지 - timesteps에 나누어떨어지는 값으로 할것
gpu = 1                    #gpu 쓸지
repeat = None              #반복해서 sampling 할지

transform = Compose([
            #transforms.CenterCrop(256),
            transforms.Resize((image_size,image_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2) - 1),
])

# define function 


def transforms(examples):   

   examples["pixel_values"] = [transform(image) for image in examples["image"]]

   del examples["image"]

   return examples

print(dataset)

transformed_dataset = dataset.with_transform(transforms)

print(transformed_dataset["train"])

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
""" dataiter = iter(dataloader)
plt.imshow(np.fliplr(np.rot90(np.transpose(next(dataiter)['pixel_values'][0]/2+0.5), 3)))
plt.show() """

if training_state == 0:
   repeat = int(timesteps/dt)
else:
   repeat = 1


for i in range(repeat):
   if gpu:
      device = "cuda" if torch.cuda.is_available() else "cpu"
   else:
      device = "cpu"

   print(device)



   trainer = Trainer(device, dataloader, image_size, channels, batch_size, timesteps, epochs, model_name)

   if training_state:
      trainer.training(training_continue)
   else:
      trainer.save_sample_time(check, dt)
      check = 0
   if device=="cuda":
      torch.cuda.empty_cache()
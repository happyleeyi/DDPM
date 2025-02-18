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

from Basic import num_to_groups
from Model import Unet
from Diffusion import Diffusion

class Trainer():
    def __init__(self, device, dataloader, image_size, channels, batch_size, timesteps, epochs, model_name):
        self.device = device
        self.dataloader = dataloader
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.epochs = epochs
        self.model = Unet(
            dim=self.image_size,
            channels=self.channels,
            dim_mults=(1, 2, 4,)
        )
        self.diffusion = Diffusion(self.timesteps,0.0001,0.02)
        self.model_name = model_name

    def training(self, training_continue):
        self.model.to(self.device)

        if training_continue:
            model_state_dict = torch.load(self.model_name, map_location=self.device)
            self.model.load_state_dict(model_state_dict)

        optimizer = Adam(self.model.parameters(), lr=5e-5)

        for epoch in range(self.epochs):
            for step, batch in enumerate(self.dataloader):
                optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = self.diffusion.p_losses(self.model, batch, t, loss_type="huber")

                if step % 100 == 0:
                    print("Epoch, Loss:", epoch, loss.item())

                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), self.model_name)
    def sampling(self, check, dt):
        self.model.to(self.device)
        model_state_dict = torch.load(self.model_name, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        samples = self.diffusion.sample(check, dt, self.model, image_size=self.image_size, batch_size=1, channels=self.channels)
        return samples
    def save_sample_time(self, check, dt):
        samples = self.sampling(check, dt)
        plt.imshow(np.fliplr(np.rot90(np.transpose(samples[-1][0])/2+0.5, 3)))
        np.save('samples.npy', np.array(samples))
        
        plt.savefig('fig_'+str(len(samples)))
        #plt.show()

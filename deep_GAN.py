import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,)),
])


batch_size=128
dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)


class Generator(nn.Module):

    def __init__(self,z_dim=10,im_chan=1,hidden_dim=64):
        super(Generator, self).__init__()

        self.z_dim=z_dim

        self.gen=nn.Sequential(
            self.gen_block(z_dim,hidden_dim*4),
            self.gen_block(hidden_dim*4,hidden_dim*2,kernel_size=4,stride=1),
            self.gen_block(hidden_dim*2,hidden_dim),
            self.gen_block(hidden_dim,im_chan,kernel_size=4,final_layer=True)
        )






    def gen_block(self,input_channels,output_channels,kernel_size=3,stride=2,final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self,noise):
        # noise is with(n_samples,z_dim)
        return noise.view(len(noise),self.z_dim,1,1)

    def forward(self,noise):

        x=self.unsqueeze_noise(noise)
        return self.gen(x)


    def get_model(self):
        return self.gen

    def get_noise(self,n_samples,device="cpu"):
        return torch.randn(n_samples,self.z_dim,device=device)



gen = Generator()
print(gen.get_model())
a=gen.get_noise(10)
out=gen.forward(a)
print(out.shape)

class Discriminator(nn.Module):
    
    def __init__(self,im_channels=1,hidden_dim=16):
        super(Discriminator, self).__init__()

        self.disc=nn.Sequential(
            self.disc_block(im_channels,hidden_dim),
            self.disc_block(hidden_dim,hidden_dim*2),
            self.disc_block(hidden_dim*2,1,final_layer=True)
        )


    def disc_block(self,input_channels,output_channels,kernel_size=4,stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )

        else:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size,stride=stride)
            )

    def forward(self,image):

        pred = self.disc(image)
        return pred.view(len(pred),-1)


print(out.shape)

disc=Discriminator()
out2= disc.forward(out)
print(out2.shape)

###    TRAINING

criterion= nn.BCEWithLogitsLoss()
z_dim=64
display_step=500
batch_size=128
lr=0.0002

beta1=0.5
beta2=0.999

gen1= Generator(z_dim=z_dim)
gen_opt=optim.Adam(gen1.parameters(),lr=lr,betas=(beta1,beta2))
disc1=Discriminator()
disc_opt=optim.Adam(disc1.parameters(),lr=lr,betas=(beta1,beta2))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen1 = gen1.apply(weights_init)
disc1 = disc1.apply(weights_init)

n_epochs=50

from tqdm import tqdm

for epoch in range(n_epochs):

    for real,_ in tqdm(dataloader):
        batch_size= len(real)

        disc_opt.zero_grad()
        fake_noise=gen1.get_noise(batch_size)
        gen_out = gen1.forward(fake_noise)
        disc_out_fake=disc1.forward(gen_out.detach())
        disc_loss_fake = criterion(disc_out_fake,torch.zeros_like(disc_out_fake))
        disc_out_real = disc1.forward(real)
        real_loss=criterion(disc_out_real,torch.ones_like(disc_out_real))
        loss = (disc_loss_fake+real_loss)/2
        loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()

        fn = gen1.get_noise(batch_size)
        gen_o= gen1.forward(fn)
        disc_pred=disc1.forward(gen_o)
        loss1 = criterion(disc_pred,torch.ones_like(disc_pred))
        loss1.backward()
        gen_opt.step()

    print(loss,loss1)


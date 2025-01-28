from PictureDamager_torch import PictureDamager
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import Adam
from torch import nn
import einops
import torch
import joblib
import numpy as np
import os
import streamlit as st
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_dataset
from copy import deepcopy

st.set_page_config(layout="wide")
empty_image=Image.new("RGB", (256, 256), color="black")
batch_size = 32
enable_pin_memory = True
start_idx=10
number_of_workers =  3
device='cpu' # IMPORTANT: Change to cuda if posssible

@st.cache_data
def load_huggingface_dataset():
    dataset = load_dataset("Artificio/WikiArt_Full",split=f'train[{start_idx}:{start_idx+batch_size}]')
    
    return dataset
@st.cache_data
def load_masks():
    masks_path = "../masksnpy"
    mini_MASKS = [np.load(os.path.join(masks_path, f)) for f in os.listdir(masks_path) if f.endswith(".npy")] 
    return mini_MASKS 
dataset = load_huggingface_dataset()
mini_MASKS=load_masks()

from torchvision import models
class Encoder(nn.Module):
  @staticmethod
  def ConvBlock(in_channels:int,out_channels:int):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.ReLU(True)
    )
  def __init__(self,latent_dim:int=32768) -> None:
    super().__init__()
    self.latent_dim=latent_dim
    resnet34 = models.resnet34(pretrained=True)
    self.model = nn.Sequential(
        *list(resnet34.children())[:-2],
        nn.Conv2d(512,52,kernel_size=1,stride=1,padding=0)
        )
  def forward(self,x):
    x=self.model(x)
    return x
class Decoder(nn.Module):
  @staticmethod
  def ConvBlock(in_channels:int,out_channels:int):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
  def __init__(self,latent_dim:int=32768) -> None:
    super().__init__()
    self.latent_dim=latent_dim
    self.model=nn.Sequential(
            nn.Conv2d(52,512,kernel_size=1,stride=1,padding=0),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Decoder.ConvBlock(512,256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Decoder.ConvBlock(256,128),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Decoder.ConvBlock(128,64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Decoder.ConvBlock(64,64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64,3,kernel_size=3,stride=1,padding=1),

            nn.Sigmoid()
    )

  def forward(self,x):
    x=self.model(x)
    return x
class ArtAutoEncoder(nn.Module):

    def __init__(self):
        super(ArtAutoEncoder, self).__init__()

        self.encoder=Encoder()


        self.decoder=Decoder()
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x
class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, in_channels=3, out_channels=3, embedding_dim=4):
        super(LitModel, self).__init__()

        self.model = ArtAutoEncoder()
        self.learning_rate = learning_rate
        self.conv_ = nn.Conv2d(in_channels + embedding_dim, out_channels, kernel_size=3)

        self.embedder = nn.Embedding(num_embeddings=20, embedding_dim=embedding_dim)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv_(x)
        x = self.model(x)
        return x

    def add_embedding_dims(self, x, clusters):
        embeddings = self.embedder(clusters)
        embeddings = einops.repeat(embeddings, 'batch_size embedding_dim -> batch_size embedding_dim h w', h=256, w=256)
        x_with_embeddings = torch.cat([embeddings, x], dim=1)

        return x_with_embeddings

    def step(self, batch, batch_idx):
        x = batch['image'].to(device).float()
        original_x=x=x/255.0
      

        batch_with_4th_channel = torch.stack([add_4th_channel(img) for img in x])

        x = torch.stack([damage_image(img) for img in batch_with_4th_channel])
        
     

        with torch.inference_mode():
            latents = model_for_clusters.encoder(x)

        latents= latents.view(latents.shape[0], -1)

        clusters = torch.from_numpy(get_image_cluster(latents)).to(device)
        x_with_embeddings = self.add_embedding_dims(x, clusters)


        x_recon = self(x_with_embeddings)

        loss = F.mse_loss(x_recon,  original_x)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('validation_loss', loss)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
class SuperResolutionAutoencoder(nn.Module):
  @staticmethod
  def ConvBlock(in_channels: int, out_channels: int):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.ReLU(True)
      )

  def __init__(self):
    super().__init__()
    #Encoder
    self.conv_1=SuperResolutionAutoencoder.ConvBlock(3,64)
    self.conv_2=SuperResolutionAutoencoder.ConvBlock(64,64)
    self.pool_1=nn.MaxPool2d(kernel_size=2)
    self.drop_1=nn.Dropout(0.3)

    self.conv_3=SuperResolutionAutoencoder.ConvBlock(64,128)
    self.conv_4=SuperResolutionAutoencoder.ConvBlock(128,128)
    self.pool_2=nn.MaxPool2d(kernel_size=2)
    self.conv_5=SuperResolutionAutoencoder.ConvBlock(128,256)

    #Decoder
    self.upsampling_1=nn.UpsamplingBilinear2d(scale_factor=2)
    self.conv_6=SuperResolutionAutoencoder.ConvBlock(256,128)
    self.conv_7=SuperResolutionAutoencoder.ConvBlock(128,128)

    self.upsampling_2=nn.UpsamplingBilinear2d(scale_factor=2)
    self.conv_8=SuperResolutionAutoencoder.ConvBlock(128,64)
    self.conv_9=SuperResolutionAutoencoder.ConvBlock(64,64)

    self.conv_10=nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

  def forward(self,x):
    #Encoder
    x=self.conv_1(x)
    x2=self.conv_2(x)
    x=self.pool_1(x2)
    x=self.drop_1(x)
    x=self.conv_3(x)
    x4=self.conv_4(x)
    x=self.pool_2(x4)
    x=self.conv_5(x)
    #Decoder
    x=self.upsampling_1(x)
    x=self.conv_6(x)
    x=self.conv_7(x)
    x=x+x4
    x=self.upsampling_2(x)
    x=self.conv_8(x)
    x=self.conv_9(x)
    x=x+x2
    x=self.conv_10(x)
    x=nn.Sigmoid()(x)
    return x
damage = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Resize((256, 256)),
])
class LModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(LModule, self).__init__()

        self.autoencoder=SuperResolutionAutoencoder().to(device)

        self.learning_rate = learning_rate


        self.save_hyperparameters()

    def forward(self, x):
        x=self.autoencoder(x)
        return x


    def step(self, batch, batch_idx):
        label = batch['image'].to(device).float()
        label=label/255.0
        x=damage(label)

        x_recon = self(x)

        loss = F.mse_loss(x_recon, label)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('validation_loss', loss)
        return loss
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
    
superr=LModule.load_from_checkpoint("../superresolution/model-epoch=01-validation_loss=0.00.ckpt").float()
superr.eval()



inpainting_model_path = "../modelFINAL-epoch=06-validation_loss=0.00.ckpt"
inpainting_model = LitModel.load_from_checkpoint(inpainting_model_path)
inpainting_model = inpainting_model.float()
inpainting_model.eval()


def choose_image_randomly():    
    return dataset['image'][np.random.randint(0,batch_size)]
def change_image():
    st.session_state.current_image = choose_image_randomly()
    st.session_state.damaged_image=empty_image
    st.session_state.damaged_image_4c=None
    st.session_state.inpaint_image=empty_image
    st.session_state.sr_image=empty_image
def add_4th_channel(img):
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)

    C, H, W = img.shape
    damager = PictureDamager(mini_MASKS, percentage_damage,device='cpu')

    mask = damager.generate_random_mask((H, W)) 
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).float().to(device) 

    mask = mask.unsqueeze(0)  

    img4 = torch.cat([img, mask], dim=0)
    return img4.to(device)

def damage_image():
    st.session_state.damaged_image=add_4th_channel(st.session_state.current_image)
    st.session_state.damaged_image_4c=st.session_state.damaged_image
    rgb = st.session_state.damaged_image[:3]  
    mask = st.session_state.damaged_image[3]  

    mask = mask.round() 

    rgb_damaged = rgb.clone().to(device)  
    rgb_damaged[:, mask == 1] = 1.0

    st.session_state.damaged_image=transforms.ToPILImage()(rgb_damaged)
    
kmeans = joblib.load("../kmeans_20_clusters.pkl")
def get_image_cluster(latents):
    latents = latents.cpu().numpy() 
    return kmeans.predict(latents)
model_for_clusters=ArtAutoEncoder().to(device)
name="resnet34_autoencoder"
model_for_clusters.load_state_dict(torch.load(f"../autoencoder/{name}.pth",map_location=device))
model_for_clusters=model_for_clusters.float()

model_for_clusters.eval()

def inpaint(original_img, damage_mask, reconstructed_img):

    damage_mask = damage_mask.expand_as(original_img)

    inpainted_img = torch.where(damage_mask > 0.5, reconstructed_img, original_img)

    return inpainted_img

def inpaint_image():
    img=transforms.ToTensor()(st.session_state.damaged_image)
    img=img.float()
    with torch.inference_mode():
        latent = model_for_clusters.encoder(img.unsqueeze(0))
      
    latent = latent.view(latent.shape[0], -1)

    clusters = torch.from_numpy(get_image_cluster(latent)).to(device)  
    images_with_embeddings = inpainting_model.add_embedding_dims(img.unsqueeze(0), clusters)
    reconstructed=inpainting_model(images_with_embeddings).squeeze(0)
    print(torch.histogram(transforms.ToTensor()(st.session_state.current_image)))
    damage_mask=st.session_state.damaged_image_4c[3:4]

    inpainted=reconstructed
    inpainted=inpaint(transforms.ToTensor()(st.session_state.current_image),damage_mask,reconstructed)
    st.session_state.inpaint_image=transforms.ToPILImage()(inpainted)
    
def sr_image():
    t=transforms.ToTensor()(st.session_state.inpaint_image)
    st.session_state.sr_image=transforms.ToPILImage()(superr(t.unsqueeze(0)).squeeze(0))
        

    
      
    
    
percentage_damage=st.slider("Damage factor",min_value=0.04,max_value=0.06,step=0.001,format="%4f",on_change=lambda:damage_image())

if "current_image" not in st.session_state:
    st.session_state.current_image = choose_image_randomly()
    st.session_state.damaged_image=empty_image
    st.session_state.inpaint_image=empty_image
    st.session_state.sr_image=empty_image
    st.session_state.damaged_image_4c=None



st.title("Inpainting demo")

image_container = st.container()
with image_container:
    st.write("")  
    col1, col2 = st.columns([1, 1], gap="large")  

    with col1:
        st.image(st.session_state.current_image, caption="Image", use_container_width=True,)
        st.image(st.session_state.inpaint_image, caption="Inpaint", use_container_width=True)

    with col2:
        st.image(st.session_state.damaged_image, caption="Damaged", use_container_width=True)
        st.image(st.session_state.sr_image, caption="Superresolution", use_container_width=True)

# Center-align the buttons
button_container = st.container()
with button_container:
    st.write("")  # Empty line for spacing
    button_col1, button_col2, button_col3,button_col4 = st.columns([1, 1, 1,1], gap="large")

    with button_col1:
        if st.button("Random photo", use_container_width=True):
            change_image()
            
    with button_col2:
        if st.button("Damage", use_container_width=True):
            damage_image()
        
    with button_col3:
        if st.button("Inpaint", use_container_width=True):
            inpaint_image()
    with button_col4:
        if st.button("SuperResolution", use_container_width=True):
            sr_image()
        


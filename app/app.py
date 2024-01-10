import streamlit as st
from torch.utils.data import DataLoader
from model import PetModel
from load_data import load_test_data 
from compute import predict
import os
import tempfile
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import torch

st.title("Agriculture Land Parcel Delineation")

nc_file = st.file_uploader("Choose Sentinel-2 satellite image (.nc file)")
tif_file = st.file_uploader("Choose boundary file (.tif file)")
nc_file_path = None
tif_file_path = None
temp_dir = tempfile.TemporaryDirectory()
if nc_file is not None:
    nc_file_path = os.path.join(temp_dir.name, nc_file.name)
    with open(nc_file_path,"wb") as f: 
        f.write(nc_file.getbuffer())  
if tif_file is not None:
    tif_file_path = os.path.join(temp_dir.name, tif_file.name)
    with open(tif_file_path,"wb") as f: 
        f.write(tif_file.getbuffer())  

button = st.button("Run")

f1 = st.empty()
f2 = st.empty()

def show_raw_image():
    fig, ax = plt.subplots(figsize=(3,3.2))
    plt.title('{}\n NIR band (B8)'.format(nc_file.name))
    ax.plot()
    imp_mpl = plt.imshow(test_dataset[0]['image'][2,:,:],cmap='gray')
    ax.add_image(imp_mpl)
    ax.axis('off')
    tmpfile_name = os.path.join(temp_dir.name, "tmpfile.png")
    fig.savefig(tmpfile_name)
    fig_image = Image.open(tmpfile_name)
    f1.image(fig_image)


def plot_result(batch, pr_masks):
    cnt = 0
    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        fig, ax = plt.subplots(figsize=(10,4))
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy()[2,:,:],cmap='gray')  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")
    
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")
    
        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        tmpfile_name = os.path.join(temp_dir.name, "tmpfile"+str(cnt)+".png")
        fig.savefig(tmpfile_name)
        fig_image = Image.open(tmpfile_name)
        f2.image(fig_image)

        cnt+=1


if button:
    test_dataset = load_test_data(nc_file_path, tif_file_path)
    BATCH_SIZE = 16
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_folder = '/Users/shohei/ghq/huggingface.co/shoheiaoki/delineation-satellite/'
    model_path = model_folder+'model_500epochs.pth'
    model = PetModel("Unet", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(torch.load(model_path))
    
    batch = next(iter(test_dataloader))
    
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    plot_result(batch, pr_masks)
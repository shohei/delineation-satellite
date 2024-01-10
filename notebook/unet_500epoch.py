#!/usr/bin/env python
# coding: utf-8

# To run the script
# $ nohup python script.py  > nohup.txt &
# $ tail -f nohup.txt 


import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader
import glob 
from osgeo import gdal
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from tqdm import tqdm
import pandas as pd
train_files = glob.glob('/home/shohei/sentinel/train/*.nc')
val_files = glob.glob('/home/shohei/sentinel/val/*.nc')
test_files = glob.glob('/home/shohei/sentinel/test/*.nc')

TRAIN_LENGTH = len(train_files)
# TRAIN_LENGTH = 1000
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

print(TRAIN_LENGTH)
print('train image')

train_dataset2 = []
valid_dataset2 = []
test_dataset2 = []

for i,file in tqdm(enumerate(train_files)):
    if i>TRAIN_LENGTH:
        break
    base_file = file.split('.nc')[0].split('train/')[1]
    data = Dataset('/home/shohei/sentinel/train/'+base_file+'.nc')
    fp = '/home/shohei/sentinel/masks/train/'+base_file+'.tif'
    img = rasterio.open(fp)
    time_slice = 0 # 0 to 5
    B2 = np.array(data['B2'][time_slice,:,:])
    B3 = np.array(data['B3'][time_slice,:,:])
    B4 = np.array(data['B4'][time_slice,:,:])
    B8 = np.array(data['B8'][time_slice,:,:])
    B2 = (255 * (B2 / B2.max() )).astype(np.int8)
    B3 = (255 * (B3 / B3.max() )).astype(np.int8)
    B4 = (255 * (B4 / B4.max() )).astype(np.int8)
    B8 = (255 * (B8 / B8.max() )).astype(np.int8)

    extent_img = img.read(1)
    boundary_img = img.read(2)
    distance_img = img.read(3)
    extent_img = extent_img.astype(np.uint8)
    boundary_img = boundary_img.astype(np.uint8)
    distance_img = distance_img.astype(np.uint8)
        
    train_input = np.zeros((3,256,256))
    train_mask  = np.zeros((1,256,256))

    # train_input[0,:,:] = B2
    train_input[0,:,:] = B3
    train_input[1,:,:] = B4
    train_input[2,:,:] = B8
    train_mask[0,:,:] = boundary_img
    # train_mask[:,:,1] = boundary_img
    # train_mask[:,:,2] = distance_img

    train_dataset2.append({'image': train_input.astype(np.uint8), 'mask': train_mask.astype(np.float32)})

print('valid image')

for i,file in tqdm(enumerate(val_files)):
    if i>TRAIN_LENGTH//2:
        break
    base_file = file.split('.nc')[0].split('val/')[1]
    data = Dataset('/home/shohei/sentinel/val/'+base_file+'.nc')
    fp = '/home/shohei/sentinel/masks/val/'+base_file+'.tif'
    img = rasterio.open(fp)
    time_slice = 0 # 0 to 5
    B2 = np.array(data['B2'][time_slice,:,:])
    B3 = np.array(data['B3'][time_slice,:,:])
    B4 = np.array(data['B4'][time_slice,:,:])
    B8 = np.array(data['B8'][time_slice,:,:])
    B2 = (255 * (B2 / B2.max() )).astype(np.int8)
    B3 = (255 * (B3 / B3.max() )).astype(np.int8)
    B4 = (255 * (B4 / B4.max() )).astype(np.int8)
    B8 = (255 * (B8 / B8.max() )).astype(np.int8)

    extent_img = img.read(1)
    boundary_img = img.read(2)
    distance_img = img.read(3)
    extent_img = extent_img.astype(np.uint8)
    boundary_img = boundary_img.astype(np.uint8)
    distance_img = distance_img.astype(np.uint8)
        
    # train_input = np.zeros((256,256,4))
    val_input = np.zeros((3,256,256))
    val_mask  = np.zeros((1,256,256))

    # train_input[0,:,:] = B2
    val_input[0,:,:] = B3
    val_input[1,:,:] = B4
    val_input[2,:,:] = B8
    val_mask[0,:,:] = boundary_img
    # train_mask[:,:,1] = boundary_img
    # train_mask[:,:,2] = distance_img

    valid_dataset2.append({'image': val_input.astype(np.uint8), 'mask': val_mask.astype(np.float32)})

print('test image')

for i,file in tqdm(enumerate(test_files)):
    if i>TRAIN_LENGTH//2:
        break
    base_file = file.split('.nc')[0].split('test/')[1]
    data = Dataset('/home/shohei/sentinel/test/'+base_file+'.nc')
    fp = '/home/shohei/sentinel/masks/test/'+base_file+'.tif'
    img = rasterio.open(fp)
    time_slice = 0 # 0 to 5
    B2 = np.array(data['B2'][time_slice,:,:])
    B3 = np.array(data['B3'][time_slice,:,:])
    B4 = np.array(data['B4'][time_slice,:,:])
    B8 = np.array(data['B8'][time_slice,:,:])
    B2 = (255 * (B2 / B2.max() )).astype(np.int8)
    B3 = (255 * (B3 / B3.max() )).astype(np.int8)
    B4 = (255 * (B4 / B4.max() )).astype(np.int8)
    B8 = (255 * (B8 / B8.max() )).astype(np.int8)

    extent_img = img.read(1)
    boundary_img = img.read(2)
    distance_img = img.read(3)
    extent_img = extent_img.astype(np.uint8)
    boundary_img = boundary_img.astype(np.uint8)
    distance_img = distance_img.astype(np.uint8)
        
    test_input = np.zeros((3,256,256))
    test_mask  = np.zeros((1,256,256))

    # test_input[0,:,:] = B2
    test_input[0,:,:] = B3
    test_input[1,:,:] = B4
    test_input[2,:,:] = B8
    test_mask[0,:,:] = boundary_img
    # test_mask[:,:,1] = boundary_img
    # test_mask[:,:,2] = distance_img
    
    test_dataset2.append({'image': test_input.astype(np.uint8), 'mask': test_mask.astype(np.float32)})

train_dataset = train_dataset2
valid_dataset = valid_dataset2
test_dataset = test_dataset2

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=n_cpu)

class PetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


model = PetModel("Unet", "resnet34", in_channels=3, out_classes=1)

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=500,
)

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)
# run test dataset
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

model_path = 'model_500epochs.pth'
torch.save(model.state_dict(), model_path)

# batch = next(iter(test_dataloader))
# with torch.no_grad():
#     model.eval()
#     logits = model(batch["image"])
# pr_masks = logits.sigmoid()

# for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 3, 1)
#     # plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
#     plt.imshow(image.numpy()[2,:,:],cmap='gray')  # convert CHW -> HWC
#     plt.title("Image")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
#     plt.title("Ground truth")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
#     plt.title("Prediction")
#     plt.axis("off")

#     plt.show()


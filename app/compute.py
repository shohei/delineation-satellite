import torch
import matplotlib.pyplot as plt 

def predict(test_dataloader, model, batch):
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()
    
    for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
        plt.figure(figsize=(10, 5))
    
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
    
    
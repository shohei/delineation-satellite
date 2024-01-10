import matplotlib.pyplot as plt
from rasterio.plot import show
from torch.utils.data import DataLoader
import torch
from model import PetModel
from load_data import load_test_data 
from compute import predict

if __name__=="__main__":
    base_file = 'AT_316_S2_10m_256'
    data_file = './app/data/'+base_file+'.nc'
    boundary_file = './app/data/'+base_file+'.tif'
    test_dataset = load_test_data(data_file, boundary_file)
    BATCH_SIZE = 16
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    plt.imshow(test_dataset[0]['image'][2,:,:],cmap='gray')
    plt.title("{0}: NIR band (B8):".format(base_file))
    plt.axis('off')

    test_dataset = load_test_data(base_file)
    BATCH_SIZE = 16
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    plt.imshow(test_dataset[0]['image'][2,:,:],cmap='gray')
    plt.title("{0}: NIR band (B8):".format(base_file))
    plt.axis('off')
    
    model_folder = '/Users/shohei/ghq/huggingface.co/shoheiaoki/delineation-satellite/'
    model_path = model_folder+'model_500epochs.pth'
    model = PetModel("Unet", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(torch.load(model_path))
    
    batch = next(iter(test_dataloader))
    
    predict(test_dataloader, model, batch)
    
    plt.show()
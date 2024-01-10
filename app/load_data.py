import numpy as np
from netCDF4 import Dataset
import rasterio

def load_test_data(data_file, boundary_file):
    data = Dataset(data_file)
    fp = boundary_file
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
    
    test_input[0,:,:] = B3
    test_input[1,:,:] = B4
    test_input[2,:,:] = B8
    test_mask[0,:,:] = boundary_img
    
    test_dataset = [{'image': test_input.astype(np.uint8), 'mask': test_mask.astype(np.float32)}]
    return test_dataset
    
    
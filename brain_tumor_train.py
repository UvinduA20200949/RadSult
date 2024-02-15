from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from brain_tuomor_preporcess import prepare
from brain_tumor_utilities import train


data_dir = ''
model_dir = '' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:0")
model = UNet(
    dimensions=3, # do 3d segmentation
    in_channels=1, #since input has only 1 chanel
    out_channels=2, #1st chanel - pixel prob of back ground,  2nd chanel-pixel probability of foreground
    channels=(16, 32, 64, 128, 256), #control how many filters in convolutional blocks
    strides=(2, 2, 2, 2),
    num_res_units=2,#calculate residuals
    norm=Norm.BATCH,#calculate BATCH
).to(device)


loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 600, model_dir)

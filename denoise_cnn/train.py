from __future__ import print_function
from math import log10
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import DenoiseCNN
from data import get_dataset

def train(epoch):
    epoch_loss = 0
    for iteration, (input, target) in enumerate(training_data_loader, 1):
        input, target = Variable(input).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        prediction = model(input)
        loss = criterion(prediction, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(training_data_loader)))
    scheduler.step(epoch_loss / len(training_data_loader))
    return input, prediction

def validate():
    avg_psnr = 0
    avg_loss = 0
    for (input, target) in testing_data_loader:
        input, target = Variable(input).cuda(), Variable(target).cuda()
        prediction = model(input)
        mse = criterion(prediction, target)
        avg_loss += mse.data[0]
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr

    print("===> Avg. Loss: {:.7f}, Avg. PSNR: {:.4f} dB".format(avg_loss/len(testing_data_loader), avg_psnr / len(testing_data_loader)))
    return input, prediction, target

def test(model, boost_tensor):
    # preprocess
    #boost_tensor[:, :, :3] = torch.clamp(boost_tensor[:, :, :3], 0, 1)
    boost_tensor[:, :, 9] /=  torch.max(boost_tensor[:, :, 9])
    boost_tensor[:, :, 10] /=  torch.max(boost_tensor[:, :, 10])
    boost_tensor[:, :, 11] /=  torch.max(boost_tensor[:, :, 11])
    boost_tensor[:, :, 12] /=  torch.max(boost_tensor[:, :, 12])
    boost_tensor[:, :, 13] /=  torch.max(boost_tensor[:, :, 13])
    # convert from (512, 512, 14) to (1, 14, 512, 512)
    boost_tensor = torch.unsqueeze(boost_tensor, 0)
    boost_tensor = boost_tensor.permute(0, 3, 1, 2)
    boost_tensor = torch.autograd.Variable(boost_tensor)
    # denoise
    output = model(boost_tensor)
    # transform back to (512, 512, 3)
    output = output.permute(0, 2, 3, 1)
    output = output.data
    return output[0]

def checkpoint(base_dir):
    model_out_path = "{}/model_epoch.pth".format(base_dir)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def load_pretrained():
    cnn = torch.load("/home/moejoe/cs240a/cuda-pathtrace/denoise_cnn/results/12553296/model_epoch.pth").cuda()
    cnn.eval()
    return cnn

if __name__ == "__main__":
    print('===> Loading datasets')
    train_set, test_set = get_dataset()
    training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=2, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    print('===> Building model')
    model = DenoiseCNN().cuda()
    #criterion = nn.L1Loss().cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=500, verbose=True, threshold=1e-4)

    time_str = str(int(time.time()))[2::]
    base_dir = "results/" + time_str

    for epoch in range(1, 400001):
        input, prediction = train(epoch)
        if epoch % 50 == 0:
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            _, prediction, target = validate()
            prediction = np.swapaxes(prediction.cpu().data.numpy()[0], 0, 2)
            target = np.swapaxes(target.cpu().data.numpy()[0], 0, 2)
            plt.imsave("{}/{}_gt.png".format(base_dir, epoch), np.clip(target[:, :, :3], 0, 1))
            plt.imsave("{}/{}_out.png".format(base_dir, epoch), np.clip(prediction, 0, 1))
            checkpoint(base_dir)
        #validate()
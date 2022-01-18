import pickle
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging

class MyDataset(Dataset):
    def __init__(self, mode="train"):
        Data = np.array(torchvision.datasets.MNIST("MNIST/processed/training.pt", download=True))
        if mode == "train":
            self.trainData = Data[:50000]
        else:
            self.trainData = Data[50000:]

    def __getitem__(self, index):
        # self.trainData[index]有兩個資料，[0]代表PIL影像，[1]代表標記資料。
        # 訓練模型時使用的資料沒有經過正規化，向素質介於0、255之間。
        img = np.array(self.trainData[index][0])
        img = torch.Tensor(img).reshape((1, *img.shape))
        lbl = torch.zeros(10)
        lbl[self.trainData[index][1]] = 1.0
        return img, lbl

    def __len__(self):
        return len(self.trainData)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 8, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 6, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(6, 4, kernel_size=3, stride=1)
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1):
        x = self.relu(self.conv1(x1))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc(x)
        x = self.softmax(x)

        return x

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset_train = MyDataset(mode="train")
        dataset_val = MyDataset(mode="val")
        self.dataloader_train = DataLoader(dataset_train, batch_size=128)
        self.dataloader_val = DataLoader(dataset_val, batch_size=100)
        self.model = Network().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def train(self):
        loss = None
        self.model.train()
        for batch_num, (data, label) in enumerate(self.dataloader_train):
            data = data.to(self.device)
            label = label.to(self.device)
            prediction = self.model(data)
            loss = self.loss_fn(prediction, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return torch.mean(loss)

    def val(self):
        self.model.eval()
        losses = list()
        correct_nums = list()
        with torch.no_grad():
            for image, label in self.dataloader_val:
                image = image.to(self.device)
                label = label.to(self.device)
                prediction = self.model(image)
                predicted_value = torch.argmax(prediction, dim=1)
                label_value = torch.argmax(label, dim=1)
                correct_nums.append((predicted_value == label_value).sum().cpu().data)
                losses.append(torch.mean(self.loss_fn(prediction, label)).cpu().data)
            losses = np.array(losses)
            average_loss = losses.mean()
            correct_nums = np.array(correct_nums).sum()
            correct_rate = correct_nums / 100
        return average_loss, correct_rate

    def saveModel(self, epoch, loss):
        if not os.path.isdir("./Model"):
            os.mkdir("./Model")
        files = os.listdir("./Model")
        for file in files:
            os.remove("./Model/{}".format(file))
        filename = "./Model/BestModel_{}.pickle".format(loss)
        with open(filename, 'wb') as f:
            pickle.dump(self.model.cpu().state_dict(), f)
        self.model.to(self.device)
        # torch.save(self.model.state_dict(), "./Model/{}_{:.2f}.pth".format(epoch, accuracy))


def train():
    logging.basicConfig(level=logging.DEBUG) 
    # writer = SummaryWriter()
    trainer = Trainer()
    lowestLoss = 1000000
    for epoch in range(30):
        logging.info("======= Epoch {} ==============".format(epoch))
        train_loss = trainer.train()
        logging.info("Training loss: {}".format(train_loss))
        val_loss, accuracy = trainer.val()
        logging.info("Validation loss: {}".format(val_loss))
        logging.info("Validation accuracy: {}%\n".format(accuracy))
        # writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Loss/val", val_loss, epoch)
        # writer.add_scalar("Accuracy/val", accuracy, epoch)
        # if val_loss < lowestLoss:
        #     lowestLoss = val_loss
        #     trainer.saveModel(epoch=epoch, loss=val_loss)
    # writer.close()

if __name__ == "__main__":
    train()
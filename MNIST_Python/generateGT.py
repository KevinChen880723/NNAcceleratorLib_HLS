import pickle
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset

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
        generateTest = True
        if generateTest:
            folder_name = "./hlsTest_float_folder"
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)

            with open(os.path.join(folder_name, "input.txt"), "w") as file:
                for data in x1[0].view(x1.shape[1]*x1.shape[2]*x1.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.conv1(x1)
        if generateTest:
            print(x)
            with open(os.path.join(folder_name, "conv1_output.txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.relu(x)
        if generateTest:
            with open(os.path.join(folder_name, "relu(conv1_output).txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.relu(self.conv2(x))
        if generateTest:
            with open(os.path.join(folder_name, "relu(conv2_output).txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.pool(x)
        if generateTest:
            with open(os.path.join(folder_name, "pool1_output.txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.relu(self.conv3(x))
        if generateTest:
            with open(os.path.join(folder_name, "relu(conv3_output).txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.relu(self.conv4(x))
        if generateTest:
            with open(os.path.join(folder_name, "relu(conv4_output).txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.pool(x)
        if generateTest:
            with open(os.path.join(folder_name, "pool2_output.txt"), "w") as file:
                for data in x[0].view(x.shape[1]*x.shape[2]*x.shape[3]):
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc(x)
        if generateTest:
            with open(os.path.join(folder_name, "fc_output.txt"), "w") as file:
                for data in x[0]:
                    file.write("{}\n".format(float(data.data)))
            file.close()

        x = self.softmax(x)
        return x

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset_train = MyDataset(mode="train")
        self.dataloader_train = DataLoader(dataset_train, batch_size=2)
        self.model = Network().to(self.device)
        with open("./Model/BestModel.pickle", 'rb') as f:
            weights = pickle.load(f)
            self.model.load_state_dict(weights)

    def generate(self):
        self.model.eval()
        with torch.no_grad():
            for (data, label) in self.dataloader_train:
                data = data.to(self.device)
                label = label.to(self.device)
                prediction = self.model(data)
                break

def train():
    trainer = Trainer()
    trainer.generate()

if __name__ == "__main__":
    train()
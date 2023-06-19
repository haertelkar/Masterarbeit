from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch import flatten, round, Tensor

class cnn(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(cnn, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=300,
            kernel_size=(3, 3))
        self.relu = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=300, out_channels=500,
            kernel_size=(5, 5))
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = Conv2d(in_channels=500, out_channels=1000,
        kernel_size=(2, 2))
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.conv4 = Conv2d(in_channels=300, out_channels=400,
        # kernel_size=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=16000, out_features=5000)

        self.fc2 = Linear(in_features=5000, out_features=500)
        # initialize our softmax classifier
        self.fc3 = Linear(in_features=500, out_features=classes)
    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        output[:1].round()
        output[-1:].round()
	
        return output

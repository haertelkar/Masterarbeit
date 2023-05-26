from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Tanh
from torch import flatten, cat

class znn(Module):
    def __init__(self, inFeatures, outFeatures):
        # call the parent constructor
        super(znn, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.fc1 = Linear(in_features=inFeatures, out_features=2000)
        self.relu = ReLU()
        # initialize second set of CONV => RELU => POOL layers
        self.fc15 = Linear(in_features=2000, out_features=4000)


        self.fc2 = Linear(in_features=4000, out_features=8000)


        self.fcmiddle = Linear(in_features=8000, out_features=8000)

        # initialize first (and only) set of FC => RELU layers
        self.fc3 = Linear(in_features=8000, out_features=4000)

        # initialize our softmax classifier
        self.fc4 = Linear(in_features=4000, out_features=1000)

        self.fc5 = Linear(in_features=1000, out_features=outFeatures)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc15(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fcmiddle(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        output = self.fc5(x)
	
        return output
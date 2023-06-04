from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Tanh
from torch import flatten, cat

class znnBottleneck(Module):
    def __init__(self, inFeatures, outFeatures):
        # call the parent constructor
        super(znnBottleneck, self).__init__()
        # first set fc -> ReLU -> Bottleneck -> ReLU
        self.fc1 = Linear(in_features=inFeatures, out_features=2000)
        self.relu = ReLU()
        self.fc1B = Linear(in_features=2000, out_features=200)
        # initialize second set, third set etc.
        self.fc15 = Linear(in_features=200, out_features=4000)
        self.fc15B = Linear(in_features=4000, out_features=400)

        self.fc2 = Linear(in_features=400, out_features=8000)
        self.fc2B = Linear(in_features=8000, out_features=800)

        #init middle layers
        self.fcmiddle = Linear(in_features=800, out_features=8000)
        self.fcmiddleB = Linear(in_features=8000, out_features = 800)

        # initialize last sets
        self.fc3 = Linear(in_features=800, out_features=4000)
        self.fc3B = Linear(in_features=4000, out_features=400)

        self.fc4 = Linear(in_features=400, out_features=1000)
        self.fc4B = Linear(in_features=1000, out_features=100)

        self.fc5 = Linear(in_features=100, out_features=outFeatures)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1B(x)
        x = self.relu(x)

        x = self.fc15(x)
        x = self.relu(x)
        x = self.fc15B(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc2B(x)
        x = self.relu(x)

        x = self.fcmiddle(x)
        x = self.relu(x)
        x = self.fcmiddleB(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc3B(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc4B(x)
        x = self.relu(x)

        output = self.fc5(x)
	
        return output
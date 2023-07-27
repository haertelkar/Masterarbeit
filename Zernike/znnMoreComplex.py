from torch.nn import Module, LogSoftmax, Linear, ReLU, Sequential
from torch import flatten, cat

class znn(Module):
    def __init__(self, inFeatures, outFeatures):
        # call the parent constructor
        super(znn, self).__init__()

        self.fc1 = Linear(in_features=inFeatures, out_features=2000)
        self.relu = ReLU()

        self.fc15 = Linear(in_features=2000, out_features=4000)


        self.fc2 = Linear(in_features=4000, out_features=8000)

        layers = []

        layers.append(Linear(8000, 80000))
        layers.append(self.relu)
        layers.append(Linear(80000, 80000))
        layers.append(self.relu)
        layers.append(Linear(80000, 8000))
        layers.append(self.relu)

        # Create the sequential container with all the layers
        self.fcmiddle = Sequential(*layers) 

        self.fc3 = Linear(in_features=8000, out_features=4000)

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
        # output[:1].round()
        # output[-1:].round()
	
        return output
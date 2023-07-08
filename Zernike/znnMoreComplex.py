from torch.nn import Module, LogSoftmax, Linear, ReLU, Sequential
from torch import flatten, cat

class znn(Module):
    def __init__(self, inFeatures, outFeatures):
        # call the parent constructor
        super(znn, self).__init__()

        self.fc1 = Linear(in_features=inFeatures, out_features=200)
        self.relu = ReLU()

        self.fc15 = Linear(in_features=200, out_features=400)


        self.fc2 = Linear(in_features=400, out_features=800)

        layers = []

        for i in range(5):
            layers.append(Linear(800, 800))
            layers.append(self.relu)

        # Create the sequential container with all the layers
        self.fcmiddle = Sequential(*layers) 

        self.fc3 = Linear(in_features=800, out_features=400)

        self.fc4 = Linear(in_features=400, out_features=100)

        self.fc5 = Linear(in_features=100, out_features=outFeatures)
        
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
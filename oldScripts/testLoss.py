import random
import numpy as np
import torch


labelPred= torch.randint(0,15,(3,20))

print(labelPred)
print(labelPred.reshape(-1,10,2))


# world = torch.zeros((2,15,15))

# xCoords = np.random.randint(0,15,10)
# yCoords = np.random.randint(0,15,10)

# world[0,xCoords, yCoords] = 1

# xCoords = np.random.randint(0,15,15)
# yCoords = np.random.randint(0,15,15)

# world[1,xCoords, yCoords] = 1
# world = world.flatten(start_dim=1)
# world = world.reshape((-1,15,15))





# gtCoords = torch.stack(torch.where(world == 1), 1)
# #go through batches and find element that is closest to prediction
# predTensor = torch.randint(0,15,(2,10,2))
# listOfCoords = []
# for i in range(2):
#     gtC = gtCoords[gtCoords[:,0] == i][:,1:]
#     pC = predTensor[i]
#     listOfCoordsSingleBatch = []
#     for pcSingle in pC:
#         listOfCoordsSingleBatch.append(gtC[torch.argmin(torch.sum(torch.abs(gtC-pcSingle)**2, dim = 1 , keepdim = True))])
#     # print(gtC)
#     # print(pC)
#     listOfCoords.append(torch.stack(listOfCoordsSingleBatch,0))

# print(torch.stack(listOfCoords,0))



# print(gtCoords)

# print(predTensor)
# torch.subtract(input = gtCoords, other = predTensor)
# print(torch.abs(gtCoords-predTensor))
# print(torch.sqrt(torch.sum(torch.abs(gtCoords-predTensor)**2, dim = 2, keepdim = True)))
# print(torch.sqrt(torch.sum(torch.abs(gtCoords-predTensor)**2, dim = 2, keepdim = True))[torch.argmin(torch.sum(torch.abs(gtCoords-predTensor)**2, dim = 2 , keepdim = True))])




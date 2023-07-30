import csv
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
import sys
import tempfile
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import builtins
import faulthandler
import signal
from datasets import ptychographicData

faulthandler.register(signal.SIGUSR1.value)

# sys.stdout = open("stdout.txt", "w", buffering=1)
# def print(text):
#     builtins.print(text)
#     os.fsync(sys.stdout)

# print("This is immediately written to stdout.txt")

def get_master_addr():
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    return hostnames.split()[0].decode('utf-8')

def setup(rank:int, world_size:int):   
	dist.init_process_group("nccl")

from torch.utils.data.distributed import DistributedSampler
def dataLoaderCustomized(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=10, shuffle = False):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, sampler=sampler)
    
    return dataloader


class Learner():
	def __init__(self, rank, world_size, epochs, version, classifier = False, indicesToPredict = None):
		self.version = version
		self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Running computations on {device}")

		# define training hyperparameters
		self.INIT_LR = world_size * 4e-3
		self.BATCH_SIZE = 256
		self.EPOCHS = epochs
		# define the train and val splits
		self.TRAIN_SPLIT = 0.75
		self.VAL_SPLIT = 1 - self.TRAIN_SPLIT
		self.classifier = classifier
		self.indicesToPredict = indicesToPredict
		self.modelName= None
		self.rank = rank
		self.world_size = world_size
		# setup the process groups
		
		setup(self.rank, self.world_size)
	
	def getDataLoaders(self):
		training_data, test_data = self.dataLoader()

		print("[INFO] generating the train/validation split...")
		numTrainSamples = int(len(training_data) * self.TRAIN_SPLIT)
		numValSamples = int(len(training_data) * self.VAL_SPLIT)
		numValSamples += len(training_data) - numTrainSamples - numValSamples
		(trainData, valData) = random_split(training_data,
			[numTrainSamples, numValSamples],
			generator=torch.Generator().manual_seed(42))
		# initialize the train, validation, and test data loaders
		trainDataLoader = dataLoaderCustomized(self.rank, self.world_size, trainData, shuffle=True,
			batch_size=self.BATCH_SIZE)
		valDataLoader = dataLoaderCustomized(self.rank, self.world_size, valData, batch_size=self.BATCH_SIZE)
		testDataLoader = dataLoaderCustomized(self.rank, self.world_size, test_data, batch_size=self.BATCH_SIZE)
		# calculate steps per epoch for training and validation set
		trainSteps = len(trainDataLoader.dataset) // self.BATCH_SIZE
		valSteps = len(valDataLoader.dataset) // self.BATCH_SIZE
		return trainSteps, trainDataLoader, valDataLoader, valSteps, testDataLoader, test_data

	def dataLoader(self):
		if self.modelName == "FullPixelGridML" or self.modelName == "unet":	
			#cnn
			training_data = ptychographicData(
						os.path.abspath(os.path.join("FullPixelGridML","measurements_train","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_train")), transform=torch.as_tensor, target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
					)

			test_data = ptychographicData(
						os.path.abspath(os.path.join("FullPixelGridML","measurements_test","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_test")), transform=torch.as_tensor, target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors, shift = training_data.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
					)
		if self.modelName == "ZernikeNormal" or self.modelName == "ZernikeBottleneck" or self.modelName == "ZernikeComplex":
			#znn
			training_data = ptychographicData(
						os.path.abspath(os.path.join("Zernike","measurements_train","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_train")), transform=torch.as_tensor, target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
						)

			test_data = ptychographicData(
						os.path.abspath(os.path.join("Zernike", "measurements_test","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_test")), transform=torch.as_tensor, target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors, shift = training_data.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
					)
					
		return training_data,test_data

	def loadModel(self, trainDataLoader):
		# first element to access dimensions
		trainFeatures, trainLabels = next(iter(trainDataLoader))
		numChannels = trainFeatures.shape[1]		

		if self.modelName == "FullPixelGridML":
			from FullPixelGridML.cnn import cnn
			print("[INFO] initializing the cnn model...")
			model = cnn(
				numChannels=numChannels,
				classes=len(trainLabels[1])).to(self.device)
			
		if self.modelName == "unet":
			from FullPixelGridML.unet import unet
			print("[INFO] initializing the unet model...")
			model = unet(
				numChannels=numChannels,
				classes=len(trainLabels[1])).to(self.device)

		if self.modelName == "ZernikeBottleneck":
			from Zernike.znnBottleneck import znnBottleneck
			print("[INFO] initializing the znnBottleneck model...")
			model = znnBottleneck(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1])).to(self.device)

		if self.modelName == "ZernikeNormal":
			from Zernike.znn import znn
			print("[INFO] initializing the znn model...")
			model = znn(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1])).to(self.device)
			
		if self.modelName == "ZernikeComplex":
			from Zernike.znnMoreComplex import znn
			print("[INFO] initializing the znnComplex model...")
			model = znn(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1])).to(self.device)
		
		# wrap the model with DDP
		# device_ids tell DDP where is your model
		# output_device tells DDP where to output, in our case, it is rank
		# find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
		print(f"self.rank = {self.rank}")
		model = DDP(model)#,device_ids=[self.rank])#, output_device=self.rank, find_unused_parameters=True)

		return model

	def learn(self, modelName, leave = True):
		self.modelName = modelName
		trainSteps, trainDataLoader, valDataLoader, valSteps, testDataLoader, test_data = self.getDataLoaders()
		assert(trainSteps > 0)
		assert(valSteps > 0)
		model = self.loadModel(trainDataLoader).to(self.device)

		# initialize our optimizer and loss function
		opt = AdamW(model.parameters(), lr=self.INIT_LR)#, weight_decay=1e-5)
		if self.classifier:
			lossFn = nn.CrossEntropyLoss()
		else:
			lossFn = nn.MSELoss()
		# initialize a dictionary to store training history
		disableTQDM = True
		if self.rank == 0:
			H = {
				"train_loss": [],
				"val_loss": [],
			}
			# measure how long training is going to take
			print("[INFO] training the network...")
			startTime = time.time()
			disableTQDM = False

		H = self.trainLoop(leave, trainSteps, trainDataLoader, valDataLoader, valSteps, model, opt, lossFn, disableTQDM, H)
		if self.rank == 0:
			self.evaluater(modelName, testDataLoader, test_data, model, H, startTime)
		cleanup()

	def trainLoop(self, leave, trainSteps, trainDataLoader, valDataLoader, valSteps, model, opt, lossFn, disableTQDM, H):
		# loop over our epochs
		for e in tqdm(range(0, self.EPOCHS), leave = leave, desc= "Epoch...", disable = disableTQDM):
				trainDataLoader.sampler.set_epoch(e)
				# set the model in training mode
				model.train()
				# initialize the total training and validation loss
				totalTrainLoss = 0
				totalValLoss = 0

				# loop over the training set
				for (x, y) in tqdm(trainDataLoader, leave=False, desc = "Training...", disable = disableTQDM):
					# send the input to the device
					(x, y) = (x.to(self.device), y.to(self.device))
					# perform a forward pass and calculate the training loss
					pred = model(x)
					loss = lossFn(pred, y)
					loss.backward()
					opt.step()
					# zero out the gradients, perform the backpropagation step,
					# and update the weights
					opt.zero_grad()
					# add the loss to the total training loss so far and
					totalTrainLoss += loss
							
				# switch off autograd for evaluation
				if self.rank == 0:
					H = self.validationLoss(trainSteps, valDataLoader, valSteps, model, lossFn, H, e, totalTrainLoss, totalValLoss)
					if H["train_loss"][-1] < 1e-5 and H["val_loss"][-1] < 1e-5:
						break
		return H

	def validationLoss(self, trainSteps, valDataLoader, valSteps, model, lossFn, H, e, totalTrainLoss, totalValLoss):
		with torch.no_grad():
			# set the model in evaluation mode
			model.eval()
			# loop over the validation set
			for (x, y) in valDataLoader:
				# send the input to the device
				(x, y) = (x.to(self.device), y.to(self.device))
				# make the predictions and calculate the validation loss
				pred = model(x)
				totalValLoss += lossFn(pred, y)
				# calculate the average training and validation loss
			avgTrainLoss = totalTrainLoss / trainSteps
			avgValLoss = totalValLoss / valSteps
			# update our training history
			tqdm.write(f"epoch {e}, train loss {avgTrainLoss}, val loss {avgValLoss}")
			H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
			H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		return H

	def evaluater(self, modelName, testDataLoader, test_data :ptychographicData, model, H, startTime):
		# print the model training and validation information
		print("Finished training model.")
		print("Train loss: {:.6f}".format(H["train_loss"][-1]))
		print("Val loss: {:.6f}\n".format(H["val_loss"][-1]))

		#finish measuring how long training took
		endTime = time.time()
		print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
		# we can now evaluate the network on the test set
		print("[INFO] evaluating network...")

		# turn off autograd for testing evaluation
		with torch.no_grad(), open(f'results_{modelName}_{self.version}.csv', 'w+', newline='') as resultsTest:
			Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			if not self.classifier and self.indicesToPredict is None:
				Writer.writerow(list(test_data.columns[1:])*2)
			elif not self.classifier:
				Writer.writerow(list(test_data.columns[np.array(self.indicesToPredict)])*2)
			else:
				Writer.writerow(list(test_data.columns[np.array(self.indicesToPredict)])) 

			# set the model in evaluation mode
			model.eval()

			# loop over the test set
			for (x, y) in testDataLoader:
				# send the input to the device
				x = x.to(self.device)
				# make the predictions and add them to the list
				pred = model(x)
				if not self.classifier:
					for predEntry, yEntry in zip(pred.tolist(), y.tolist()):
						predScaled = test_data.scaleUp(row = predEntry)
						yScaled = test_data.scaleUp(row = yEntry)
						Writer.writerow(list(predScaled) +  list(yScaled))
				else:
					for predEntry,yEntry in zip(pred.cpu().numpy(), y.cpu().numpy()):
						Writer.writerow([int(predEntry.argmax() == yEntry.argmax())])
				
		self.plotTraining(H, modelName)
		torch.save(model, os.path.join("models",f"{modelName}_{self.version}.m"))

	def	plotTraining(self, H, modelName, startingPoint = 1):
		plt.figure()
		plt.plot(H["train_loss"][startingPoint:], label="train_loss")
		plt.plot(H["val_loss"][startingPoint:], label="val_loss")
		plt.title("Training Loss on Dataset")
		plt.xlabel("Epoch # - 1")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.show()
		plt.savefig(os.path.join("Loss", f'{modelName}_{self.version}.png'))
		# serialize the model to disk
		

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size,epochs, version, classifier, indicesToPredict, modelString):
	learn = Learner(rank, world_size, epochs = epochs, version = version, classifier=classifier, indicesToPredict = indicesToPredict)

	numberOfModels = 0

	for modelName in models:
		if modelString and modelString not in modelName:
			continue
		learn.learn(modelName=modelName)
		numberOfModels += 1

	if numberOfModels == 0: raise Exception("No model fits the required String.")

def getMPIWorldSize():
	return int(os.environ['OMPI_COMM_WORLD_SIZE'])

def getMPIRank():
	return int(os.environ['OMPI_COMM_WORLD_RANK'])

if __name__ == '__main__':
	models = ["FullPixelGridML", "ZernikeNormal", "ZernikeBottleneck", "ZernikeComplex"]
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--version", type=str, required=True,
	help="version number")
	ap.add_argument("-e", "--epochs", type=int, required=True,
	help="number of epochs")
	ap.add_argument("-m" ,"--models", type=str, required=False, help = "only use models with this String in their name")
	ap.add_argument("-c" ,"--classifier", type=int, required=False, default=0, help = "Use if the model is to be a classfier. Choose the label index to be classified")
	ap.add_argument("-i" ,"--indices", type=str, required=False, default="all", help = "specify indices of labels to predict (eg. '1, 2,5'). Default is all.")
	# ap.add_argument('-r', '--rank', default=-1, type=int, help='rank of the current process')
	# ap.add_argument('-ws', '--world_size', default=-1, type=int, help='number of processes participating in the job')
	args = vars(ap.parse_args())


	if not args["classifier"]:
		classifier = None
		indicesToPredict = None
	else:
		classifier = True
		indicesToPredict = args["classifier"]
		models.append("unet")

	if args["indices"] != "all":
		indices = args["indices"].split(",")
		indicesToPredict = [int(i) for i in indices]
	
	# rank = args["rank"]
	# if rank == -1: rank = getMPIRank()
	# if rank == 0: sys.stdout = open(f"stdout{rank}.txt", "w", buffering=1)
	# world_size = args["world_size"]
	# if world_size == -1: world_size = getMPIWorldSize()

	local_rank = int(os.environ["LOCAL_RANK"])
	global_rank = int(os.environ["RANK"])
	world_size = int(os.environ["WORLD_SIZE"])
	main(global_rank,world_size, args["epochs"], args["version"], classifier, indicesToPredict, args["models"])     
	
	#only for multiple gpus per node
	# mp.spawn(
    #     main,
    #     args=(world_size, args["epochs"], args["version"], classifier, indicesToPredict),
    #     nprocs=world_size
    # )
	
	

	# try:
	# 	set_start_method('spawn')
	# except RuntimeError:
	# 	pass
	# pool = Pool()
	# pool.map(learn.Learner, models)
	# pool.close()
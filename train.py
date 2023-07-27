import csv
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
import torch
from tqdm import tqdm
from datasets import ptychographicData

class Learner():
	def __init__(self, epochs, version, classifier = False, indicesToPredict = None):
		self.version = version
		self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Running computations on {device}")

		# define training hyperparameters
		self.INIT_LR = 1e-3
		self.BATCH_SIZE = 64
		self.EPOCHS = epochs
		# define the train and val splits
		self.TRAIN_SPLIT = 0.75
		self.VAL_SPLIT = 1 - self.TRAIN_SPLIT
		self.classifier = classifier
		self.indicesToPredict = indicesToPredict
		self.modelName= None
	
	def loadData(self):
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

			# training_data = multiPos(
			# 	os.path.abspath(os.path.join("Zernike","measurements_train","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_train")), transform=torch.as_tensor, target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
			# 	)

			# test_data = multiPos(
			# 	os.path.abspath(os.path.join("Zernike", "measurements_test","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_test")), transform=torch.as_tensor, target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors, shift = training_data.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
			# )

		print("[INFO] generating the train/validation split...")
		numTrainSamples = int(len(training_data) * self.TRAIN_SPLIT)
		numValSamples = int(len(training_data) * self.VAL_SPLIT)
		numValSamples += len(training_data) - numTrainSamples - numValSamples
		(trainData, valData) = random_split(training_data,
			[numTrainSamples, numValSamples],
			generator=torch.Generator().manual_seed(42))
		# initialize the train, validation, and test data loaders
		trainDataLoader = DataLoader(trainData, shuffle=True,
			batch_size=self.BATCH_SIZE)
		valDataLoader = DataLoader(valData, batch_size=self.BATCH_SIZE)
		testDataLoader = DataLoader(test_data, batch_size=self.BATCH_SIZE)
		# calculate steps per epoch for training and validation set
		trainSteps = len(trainDataLoader.dataset) // self.BATCH_SIZE
		valSteps = len(valDataLoader.dataset) // self.BATCH_SIZE
		return trainSteps, trainDataLoader, valDataLoader, valSteps, testDataLoader, test_data

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
			
		return model

	def learn(self, modelName, leave = True):
		self.modelName = modelName
		#print(f"Training model {modelName}")
		trainSteps, trainDataLoader, valDataLoader, valSteps, testDataLoader, test_data = self.loadData()
		assert(trainSteps > 0)
		assert(valSteps > 0)
		model = self.loadModel(trainDataLoader)
	
		# initialize our optimizer and loss function
		opt = AdamW(model.parameters(), lr=self.INIT_LR)#, weight_decay=1e-5)
		if self.classifier:
			lossFn = nn.CrossEntropyLoss()
		else:
			lossFn = nn.MSELoss()
		# initialize a dictionary to store training history
		H = {
			"train_loss": [],
			"val_loss": [],
		}
		# measure how long training is going to take
		print("[INFO] training the network...")
		startTime = time.time()

		# loop over our epochs
		for e in tqdm(range(0, self.EPOCHS), leave = leave, desc= "Epoch..."):
			# set the model in training mode
			model.train()
			# initialize the total training and validation loss
			totalTrainLoss = 0
			totalValLoss = 0
			# loop over the training set
			for (x, y) in tqdm(trainDataLoader, leave=False, desc = "Training..."):
				# send the input to the device
				(x, y) = (x.to(self.device), y.to(self.device))
				# perform a forward pass and calculate the training loss
				pred = model(x)
				loss = lossFn(pred, y)
				# zero out the gradients, perform the backpropagation step,
				# and update the weights
				opt.zero_grad()
				loss.backward()
				opt.step()
				# add the loss to the total training loss so far and
				totalTrainLoss += loss
						
			# switch off autograd for evaluation
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
			if avgTrainLoss < 1e-5 and avgValLoss < 1e-5:
				break
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
		print("Train loss: {:.6f}".format(avgTrainLoss))
		print("Val loss: {:.6f}\n".format(avgValLoss))

		#finish measuring how long training took
		endTime = time.time()
		print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
		# we can now evaluate the network on the test set
		print("[INFO] evaluating network...")

		self.evaluater(modelName, testDataLoader, test_data, model, H)

	def evaluater(self, modelName, testDataLoader, test_data :ptychographicData, model, H):
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
				
		self.plotTraining(H, modelName, model)

	def	plotTraining(self, H, modelName, model, startingPoint = 1):
		plt.figure()
		plt.plot(H["train_loss"][startingPoint:], label="train_loss")
		plt.plot(H["val_loss"][startingPoint:], label="val_loss")
		plt.title("Training Loss on Dataset")
		plt.xlabel("Epoch # - 1")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.show()
		plt.savefig(f'{modelName}_{self.version}.png')
		# serialize the model to disk
		torch.save(model, f"{modelName}_{self.version}.m")


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

	learn = Learner(epochs = args["epochs"], version = args["version"], classifier=classifier, indicesToPredict = indicesToPredict)

	numberOfModels = 0

	for modelName in models:
		if args["models"] and args["models"] not in modelName:
			continue
		learn.learn(modelName=modelName)
		numberOfModels += 1

	if numberOfModels == 0: raise Exception("No model fits the required String.")

	# try:
	# 	set_start_method('spawn')
	# except RuntimeError:
	# 	pass
	# pool = Pool()
	# pool.map(learn.Learner, models)
	# pool.close()
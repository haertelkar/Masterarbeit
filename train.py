import csv
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import AdamW, SGD, Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method




class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors = None, labelIndicesToPredict = None, classifier = False):
		self.img_labels = pd.read_csv(annotations_file)
		self.columns = list(self.img_labels.columns)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.labelIndicesToPredict = labelIndicesToPredict
		self.scalerZernike = 1
		assert(not(labelIndicesToPredict is 0))
		if isinstance(labelIndicesToPredict, list):
			assert(0 not in labelIndicesToPredict)
		if self.labelIndicesToPredict is None:
			self.scalingFactors = scalingFactors if scalingFactors is not None else np.array(self.img_labels.iloc[:, 1:].max(axis=0).to_list())
		elif type(self.labelIndicesToPredict) == list:
			self.scalingFactors = scalingFactors if scalingFactors is not None else np.array(self.img_labels.iloc[:, labelIndicesToPredict].max(axis=0).to_list())
		else:
			self.scalingFactors = scalingFactors if scalingFactors is not None else np.array(self.img_labels.iloc[:, labelIndicesToPredict].max(axis=0))
		self.classifier = classifier
		if self.classifier:
			assert(type(labelIndicesToPredict) != list)
			self.smallestElem = self.img_labels.iloc[:, labelIndicesToPredict].min()
			self.numberOfClasses = self.img_labels.iloc[:, labelIndicesToPredict].max() - self.smallestElem + 1
			


	def __len__(self):
		file_count = len(self.img_labels)
		return file_count

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		imageOrZernikeMoments = np.load(img_path).astype('float32')		
		if self.scalerZernike == 1 or len(imageOrZernikeMoments.shape) == 2: #only set once for Zernike
			self.scalerZernike = np.max(imageOrZernikeMoments)
		imageOrZernikeMoments /= self.scalerZernike
		if len(imageOrZernikeMoments.shape) == 2:
			imageOrZernikeMoments = imageOrZernikeMoments[:-(imageOrZernikeMoments.shape[0]%2), :-(imageOrZernikeMoments.shape[1]%2)] #only even dimensions are allowed in unet
		if self.classifier:
			label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype(int)
			label = self.createOneHotEncodedVector(label)
		else:
			label = np.array(self.img_labels.iloc[idx, 1:]).astype('float32') #gets read in as generic objects
			label = self.scaleDown(label)
		if self.target_transform:
			label = self.target_transform(label)
		if self.transform:
			imageOrZernikeMoments = self.transform(imageOrZernikeMoments) #not scaled here because it has to be datatype uint8 to be scaled automatically
		
		return imageOrZernikeMoments, label
	
	def createOneHotEncodedVector(self, classIndex):
		classes = np.zeros(self.numberOfClasses)
		classes[classIndex - self.smallestElem] = 1
		return classes

	def scaleUp(self, row):
		row = np.array(row)
		row += self.scalingFactors/2
		row *= self.scalingFactors
		return row
	
	def scaleDown(self, row):
		row = np.array(row)
		row /= self.scalingFactors + 0.00000001
		row -= self.scalingFactors/2
		return row

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
	
	def loadData(self, modelName):
		if modelName == "FullPixelGridML" or modelName == "unet":	
			#cnn
			training_data = ptychographicData(
				os.path.abspath(os.path.join("FullPixelGridML","measurements_train","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_train")), transform=ToTensor(), target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
			)

			test_data = ptychographicData(
				os.path.abspath(os.path.join("FullPixelGridML","measurements_test","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_test")), transform=ToTensor(), target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
			)
		if modelName == "Zernike" or modelName == "ZernikeBottleneck":
			#znn
			training_data = ptychographicData(
				os.path.abspath(os.path.join("Zernike","measurements_train","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_train")), transform=torch.as_tensor, target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
				)

			test_data = ptychographicData(
				os.path.abspath(os.path.join("Zernike", "measurements_test","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_test")), transform=torch.as_tensor, target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
			)

		print("[INFO] generating the train/validation split...")
		numTrainSamples = int(len(training_data) * self.TRAIN_SPLIT)
		numValSamples = int(len(training_data) * self.VAL_SPLIT)
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

		if modelName == "FullPixelGridML":
			from FullPixelGridML.cnn import cnn
			print("[INFO] initializing the cnn model...")
			model = cnn(
				numChannels=1,
				classes=len(trainLabels[1])).to(self.device)
			
		if modelName == "unet":
			from FullPixelGridML.unet import unet
			print("[INFO] initializing the unet model...")
			model = unet(
				numChannels=1,
				classes=len(trainLabels[1])).to(self.device)

		if modelName == "ZernikeBottleneck":
			from Zernike.znnBottleneck import znnBottleneck
			print("[INFO] initializing the znnBottleneck model...")
			model = znnBottleneck(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1])).to(self.device)

		if modelName == "Zernike":
			from Zernike.znn import znn
			print("[INFO] initializing the znn model...")
			model = znn(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1])).to(self.device)
			
		return model

	def Learner(self, modelName, leave = True):
		#print(f"Training model {modelName}")
		trainSteps, trainDataLoader, valDataLoader, valSteps, testDataLoader, test_data = self.loadData(modelName)
		model = self.loadModel(trainDataLoader)
	
		# initialize our optimizer and loss function
		opt = Adam(model.parameters(), lr=self.INIT_LR)#, weight_decay=1e-5)
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
			H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
			H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
		print("Train loss: {:.6f}".format(avgTrainLoss))
		print("Val loss: {:.6f}\n".format(avgValLoss))

		#finish measuring how long training took
		endTime = time.time()
		print("[INFO] total time taken to train the model: {:.2f}s".format(
			endTime - startTime))
		# we can now evaluate the network on the test set
		print("[INFO] evaluating network...")


		# turn off autograd for testing evaluation
		with torch.no_grad(), open(f'results_{modelName}_{self.version}.csv', 'w+', newline='') as resultsTest:
			Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			try:
				if not self.classifier: 
					Writer.writerow([test_data.columns[self.indicesToPredict]]*2)
				else:
					Writer.writerow([test_data.columns[self.indicesToPredict]])
			except TypeError:
				Writer.writerow(test_data.columns[1:]*2)


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
						Writer.writerow(list(predScaled) + list(yScaled))		
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
	models = ["FullPixelGridML", "Zernike", "ZernikeBottleneck"]
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--version", type=str, required=True,
	help="version number")
	ap.add_argument("-e", "--epochs", type=int, required=True,
	help="number of epochs")
	ap.add_argument("-m" ,"--models", type=str, required=False, help = "only use models with this String in their name")
	ap.add_argument("-c" ,"--classifier", type=int, required=False, default=0, help = "Use if the model is to be a classfier. Choose the label index to be classified")
	args = vars(ap.parse_args())


	if not args["classifier"]:
		classifier = None
		indicesToPredict = None
	else:
		classifier = True
		indicesToPredict = args["classifier"]

	learn = Learner(epochs = args["epochs"], version = args["version"], classifier=classifier, indicesToPredict = indicesToPredict)

	numberOfModels = 0

	for modelName in models:
		if args["models"] and args["models"] not in modelName:
			continue
		learn.Learner(modelName=modelName)
		numberOfModels += 1

	if numberOfModels == 0: raise Exception("No models fit the required String.")

	# try:
	# 	set_start_method('spawn')
	# except RuntimeError:
	# 	pass
	# pool = Pool()
	# pool.map(learn.Learner, models)
	# pool.close()
import csv
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
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
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors = None):
		self.img_labels = pd.read_csv(annotations_file)
		self.columns = list(self.img_labels.columns[1:])
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		if scalingFactors is None:
			self.scalingFactors = np.array(self.img_labels.iloc[:, 1:].max(axis=0).to_list())
		else:
			self.scalingFactors = scalingFactors


	def __len__(self):
		file_count = len(self.img_labels)
		return file_count

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = np.load(img_path).astype('float32')
		if len(image.shape) == 2:
			image = image[:-(image.shape[0]%2), :-(image.shape[1]%2)] #only even dimensions are allowed in unet
		label = np.array(self.img_labels.iloc[idx, 1:]).astype('float32') #gets read in as generic objects
		image /= np.max(image)
		if self.transform:
			image = self.transform(image) 
		label = self.scaleDown(label)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label
	
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
	def __init__(self, epochs, version):
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

	def Learner(self, modelName):
		#print(f"Training model {modelName}")

		if modelName == "FullPixelGridML":	
			#cnn
			training_data = ptychographicData(
				os.path.abspath(os.path.join("FullPixelGridML","measurements_train","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_train")), transform=ToTensor(), target_transform=torch.as_tensor
			)

			test_data = ptychographicData(
				os.path.abspath(os.path.join("FullPixelGridML","measurements_test","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML","measurements_test")), transform=ToTensor(), target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors
			)
		if modelName == "Zernike" or modelName == "ZernikeBottleneck":
			#znn
			training_data = ptychographicData(
				os.path.abspath(os.path.join("Zernike","measurements_train","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_train")), transform=torch.as_tensor, target_transform=torch.as_tensor
				)

			test_data = ptychographicData(
				os.path.abspath(os.path.join("Zernike", "measurements_test","labels.csv")), os.path.abspath(os.path.join("Zernike", "measurements_test")), transform=torch.as_tensor, target_transform=torch.as_tensor, scalingFactors = training_data.scalingFactors
			)

		print("[INFO] generating the train/validation split...")
		numTrainSamples = int(len(training_data) * self.TRAIN_SPLIT)
		numValSamples = int(len(test_data) * self.VAL_SPLIT)
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

		# initialize the model
		trainFeatures, trainLabels = next(iter(trainDataLoader))

		if modelName == "FullPixelGridML":
			from FullPixelGridML.cnn import cnn
			print("[INFO] initializing the cnn model...")
			model = cnn(
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
			
		# initialize our optimizer and loss function
		opt = Adam(model.parameters(), lr=self.INIT_LR)
		lossFn = nn.MSELoss()
		# initialize a dictionary to store training history
		H = {
			"train_loss": [],
			"train_acc": [],
			"val_loss": [],
			"val_acc": []
		}
		# measure how long training is going to take
		print("[INFO] training the network...")
		startTime = time.time()

		# loop over our epochs
		for e in tqdm(range(0, self.EPOCHS)):
			# set the model in training mode
			model.train()
			# initialize the total training and validation loss
			totalTrainLoss = 0
			totalValLoss = 0
			# initialize the number of correct predictions in the training
			# and validation step
			trainCorrect = 0
			valCorrect = 0
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
				# calculate the number of correct predictions
				totalTrainLoss += loss
				# trainCorrect += (pred.argmax(1) == y).type(
				# 	torch.float).sum().item()
						
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
					# calculate the number of correct predictions
					# valCorrect += (pred.argmax(1) == y).type(
					#     torch.float).sum().item()
				
			# calculate the average training and validation loss
			avgTrainLoss = totalTrainLoss / trainSteps
			avgValLoss = totalValLoss / valSteps
			# calculate the training and validation accuracy
			trainCorrect = trainCorrect / len(trainDataLoader.dataset)
			valCorrect = valCorrect / len(valDataLoader.dataset)
			# update our training history
			H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
			H["train_acc"].append(trainCorrect)
			H["val_loss"].append(avgValLoss.cpu().detach().numpy())
			H["val_acc"].append(valCorrect)
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, self.EPOCHS))
		print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
		print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))

		#finish measuring how long training took
		endTime = time.time()
		print("[INFO] total time taken to train the model: {:.2f}s".format(
			endTime - startTime))
		# we can now evaluate the network on the test set
		print("[INFO] evaluating network...")


		# turn off autograd for testing evaluation
		with torch.no_grad(), open('results_{modelName}_{}.csv'.format(self.version), 'w+', newline='') as resultsTest:
			Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			Writer.writerow(test_data.columns*2)
			# set the model in evaluation mode
			model.eval()
			
			# loop over the test set
			for (x, y) in testDataLoader:
				# send the input to the device
				x = x.to(self.device)
				# make the predictions and add them to the list
				pred = model(x)
				for predEntry, yEntry in zip(pred.tolist(), y.tolist()):
					predScaled = test_data.scaleUp(row = predEntry)
					yScaled = test_data.scaleUp(row = yEntry)
					Writer.writerow(list(predScaled) + list(yScaled))		

		plt.figure()
		plt.plot(H["train_loss"][1:], label="train_loss")
		plt.plot(H["val_loss"][1:], label="val_loss")
		plt.title("Training Loss on Dataset")
		plt.xlabel("Epoch # - 1")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.show()
		plt.savefig(f'{modelName}_{self.version}.png')
		# serialize the model to disk
		torch.save(model, f"{modelName}_{self.version}.m")


if __name__ == '__main__':
	models = ["FullPixelGridML", "ZernikeBottleneck", "Zernike"]
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--version", type=str, required=True,
	help="version number")
	ap.add_argument("-e", "--epochs", type=int, required=True,
	help="number of epochs")
	args = vars(ap.parse_args())

	learn = Learner(epochs = args["epochs"], version = args["version"])

	for modelName in models:
		learn.Learner(modelName=modelName)

	# try:
	# 	set_start_method('spawn')
	# except RuntimeError:
	# 	pass
	# pool = Pool()
	# pool.map(learn.Learner, models)
	# pool.close()
import csv
import warnings
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch
import faulthandler
import signal
from tqdm import tqdm
from datasets import ptychographicData
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

faulthandler.register(signal.SIGUSR1.value)


def getMPIWorldSize():
	return int(os.environ['SLURM_NNODES'])	

class lightnModelClass(pl.LightningModule):
	def __init__(self, model, lr = getMPIWorldSize() * 4e-4):
		super().__init__()
		self.model = model
		self.lr = lr

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		x, y = batch
		y_hat = self.model(x)
		loss = F.mse_loss(y_hat, y)
		return loss
	
	def validation_step(self, batch, batch_idx):
		# training_step defines the train loop.
		x, y = batch
		y_hat = self.model(x)
		loss = F.mse_loss(y_hat, y)
		self.log("val_loss", loss, sync_dist=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		#TODO go back to GPU optimizer optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		return optimizer
	
	# def train_dataloader(self):
	# 	return DataLoader(self.trainDataLoader, batch_size=self.batch_size) #maybe use functools.partial to abstract a data loader, where only the batch size is missing

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		test_loss = F.mse_loss(y_hat, y)
		self.log("test_loss", test_loss)

class Loader():
	def __init__(self, world_size, epochs, version, classifier = False, indicesToPredict = None):
		self.version = version
		self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Running computations on {device}")

		# define training hyperparameters
		self.INIT_LR = world_size * 4e-4
		self.BATCH_SIZE = 8
		self.EPOCHS = epochs
		# define the train and val splits
		self.TRAIN_SPLIT = 0.75
		self.VAL_SPLIT = 1 - self.TRAIN_SPLIT
		self.classifier = classifier
		self.indicesToPredict = indicesToPredict
		self.modelName= None
		self.world_size = world_size
		# setup the process groups
	
	def getDataLoaders(self, num_workers):
		assert(self.modelName != None)
		training_data, test_data = self.dataLoader()

		print("[INFO] generating the train/validation split...")
		numTrainSamples = int(len(training_data) * self.TRAIN_SPLIT)
		numValSamples = int(len(training_data) * self.VAL_SPLIT)
		numValSamples += len(training_data) - numTrainSamples - numValSamples
		(trainData, valData) = random_split(training_data,
			[numTrainSamples, numValSamples],
			generator=torch.Generator().manual_seed(42))
		# initialize the train, validation, and test data loaders
		warnings.filterwarnings("ignore", ".*This DataLoader will create 20 worker processes in total. Our suggested max number of worker in current system is 10.*") #this is an incorrect warning as we have 20 cores
		trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=self.BATCH_SIZE, num_workers= num_workers, pin_memory=True)
		valDataLoader = DataLoader(valData, batch_size= self.BATCH_SIZE, num_workers= num_workers, pin_memory=True)
		testDataLoader = DataLoader(test_data, batch_size= self.BATCH_SIZE, num_workers= num_workers, pin_memory=True)

		return trainDataLoader, valDataLoader, testDataLoader, test_data

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
				classes=len(trainLabels[1]))
			
		if self.modelName == "unet":
			from FullPixelGridML.unet import unet
			print("[INFO] initializing the unet model...")
			model = unet(
				numChannels=numChannels,
				classes=len(trainLabels[1]))

		if self.modelName == "ZernikeBottleneck":
			from Zernike.znnBottleneck import znnBottleneck
			print("[INFO] initializing the znnBottleneck model...")
			model = znnBottleneck(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1]))

		if self.modelName == "ZernikeNormal":
			from Zernike.znn import znn
			print("[INFO] initializing the znn model...")
			model = znn(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1]))
			
		if self.modelName == "ZernikeComplex":
			from Zernike.znnMoreComplex import znn
			print("[INFO] initializing the znnComplex model...")
			model = znn(
				inFeatures = len(trainFeatures[1]),
				outFeatures=len(trainLabels[1]))
		
		return model

	def evaluater(self, testDataLoader, test_data :ptychographicData, model):
		# we can now evaluate the network on the test set
		print("[INFO] evaluating network...")
		
		# turn off autograd for testing evaluation
		with torch.inference_mode(), open(os.path.join("testDataEval",f'results_{self.modelName}_{self.version}.csv'), 'w+', newline='') as resultsTest:
			Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			if not self.classifier and self.indicesToPredict is None:
				Writer.writerow(list(test_data.columns[1:])*2)
			elif not self.classifier:
				Writer.writerow(list(test_data.columns[np.array(self.indicesToPredict)])*2)
			else:
				Writer.writerow(list(test_data.columns[np.array(self.indicesToPredict)])) 
			
			model.eval()
			
			# loop over the test set
			for (x, y) in tqdm(testDataLoader, desc="Going through test data"):
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
	

def main(epochs, version, classifier, indicesToPredict, modelString):
	world_size = getMPIWorldSize()
	loader = Loader(world_size=world_size, epochs = epochs, version = version, classifier=classifier, indicesToPredict = indicesToPredict)
	numberOfModels = 0

	for modelName in models:
		if modelString and modelString not in modelName:
			continue
		loader.modelName = modelName
		trainDataLoader, valDataLoader, testDataLoader, test_data = loader.getDataLoaders(num_workers = 20)
		model = loader.loadModel(trainDataLoader)
		lightnModel = lightnModelClass(model)
		early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min", stopping_threshold = 1e-6)
		#StochasticWeightAveraging(swa_lrs=1e-2) faster but a bit less good
		trainer = pl.Trainer(max_epochs=epochs,num_nodes=world_size, accelerator="gpu",devices=1, callbacks=[early_stop_callback])
		
		#Create a Tuner
		tuner = Tuner(trainer)
		# finds learning rate automatically
		# sets ightnModel.lr or ightnModel.learning_rate to that learning rate
		tuner.lr_find(lightnModel, trainDataLoader, valDataLoader, max_lr = 4e-2, early_stop_threshold = 4)
		# Auto-scale batch size by growing it exponentially (default)
		# tuner.scale_batch_size(lightnModel) implement train data Loader

		trainer.fit(lightnModel, trainDataLoader, valDataLoader)
		trainer.save_checkpoint(os.path.join("models",f"{modelName}_{version}.ckpt"))
		lightnModelClass.load_from_checkpoint(checkpoint_path = os.path.join("models",f"{modelName}_{version}.ckpt"), model = model)
		loader.evaluater(testDataLoader, test_data, lightnModel)
		torch.save(lightnModel.state_dict(), os.path.join("models",f"{modelName}_{version}.m"))
		numberOfModels += 1

	if numberOfModels == 0: raise Exception("No model fits the required String.")

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
	main(args["epochs"], args["version"], classifier, indicesToPredict, args["models"])     
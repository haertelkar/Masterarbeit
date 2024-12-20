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
from FullPixelGridML.CNNFullNN import TwoPartLightningCNN
from Zernike.dqn import preCompute, TwoPartLightning
from datasets import ptychographicDataLightning
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment as SLURMEnvironment # type: ignore
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler




faulthandler.register(signal.SIGUSR1.value)


def getMPIWorldSize():
	try:
		return int(os.environ['SLURM_NNODES'])
	except KeyError: #if not on a cluster
		return 	1

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
		loss = self.lossFct(y_hat, y)
		return loss
	
	def lossFct(self, y_hat, y):
		
		return F.mse_loss(y_hat, y)

	def validation_step(self, batch, batch_idx):
		# training_step defines the train loop.
		x, y = batch
		y_hat = self.model(x)
		loss = self.lossFct(y_hat, y)
		self.log("val_loss", loss)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		return optimizer
	
	# def train_dataloader(self):
	# 	return DataLoader(self.trainDataLoader, batch_size=self.batch_size) #maybe use functools.partial to abstract a data loader, where only the batch size is missing

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		test_loss = F.mse_loss(y_hat, y)
		self.log("test_loss", test_loss)

def loadModel(trainDataLoader = None, modelName = "ZernikeNormal", numChannels = 0, numLabels = 0) -> nn.Module:
	# first element to access dimensions
	if numChannels <= 0 and numLabels <= 0 and trainDataLoader is not None: 
		trainFeatures, trainLabels = next(iter(trainDataLoader))		
		#first dim is batch size
		numChannels = trainFeatures.shape[1]		
		numLabels = trainLabels.shape[1]
	elif numChannels <= 0 or numLabels <= 0:
		raise Exception("Either numChannels or numLabels is not set correctly")

	if modelName == "FullPixelGridML":
		from FullPixelGridML.cnn import cnn
		print("[INFO] initializing the cnn model...")
		model = cnn(
			numChannels=numChannels,
			outputFeatureCount=numLabels)
		
	elif modelName == "unet":
		from FullPixelGridML.unet import unet
		print("[INFO] initializing the unet model...")
		model = unet(
			numChannels=numChannels,
			classes=numLabels)

	elif modelName == "ZernikeBottleneck":
		from Zernike.znnBottleneck import znnBottleneck
		print("[INFO] initializing the znnBottleneck model...")
		model = znnBottleneck(
			inFeatures = numChannels,
			outFeatures=numLabels)

	elif modelName == "ZernikeNormal":
		from Zernike.znn import znn
		print("[INFO] initializing the znn model...")
		model = znn(
			inFeatures = numChannels,
			outFeatures=numLabels)
		
	elif modelName == "ZernikeComplex":
		from Zernike.znnMoreComplex import znn
		print("[INFO] initializing the znnComplex model...")
		model = znn(
			inFeatures = len(trainFeatures[0]),
			outFeatures=numLabels)
		
	else:
		raise Exception(f"{modelName} is not a known model")
	
	return model

def evaluater(testDataLoader, test_data, model, indicesToPredict, modelName, version, classifier):
	# we can now evaluate the network on the test set
	print("[INFO] evaluating network...")
	model.to('cuda')

	# turn off autograd for testing evaluation
	with torch.inference_mode(), open(os.path.join("testDataEval",f'results_{modelName}_{version}.csv'), 'w+', newline='') as resultsTest:
		Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		if not classifier and indicesToPredict is None:
			predColumnNames = []
			for columnName in list(test_data.columns[1:]): 
				predColumnNames.append(f"{columnName}_pred")
			Writer.writerow(predColumnNames + list(test_data.columns[1:]))
		elif not classifier:
			predColumnNames = []
			for columnName in list(test_data.columns[np.array(indicesToPredict)]):
				predColumnNames.append(f"{columnName}_pred")
			Writer.writerow(predColumnNames+ list(test_data.columns[np.array(indicesToPredict)]))
		else:
			Writer.writerow(list(test_data.columns[np.array(indicesToPredict)])) 
		
		model.eval()
		
		
		# loop over the test set
		for (x, y) in tqdm(testDataLoader, desc="Going through test data"):

			# make the predictions and add them to the list
			pred = model(x.to('cuda'))
			y.to("cuda")
			if not classifier:
				for predEntry, yEntry in zip(pred.tolist(), y.tolist()):
					Writer.writerow(list(predEntry) +  list(yEntry))
			else:
				for predEntry,yEntry in zip(pred.cpu().numpy(), y.cpu().numpy()):
					Writer.writerow([int(predEntry.argmax() == yEntry.argmax())])
	

def main(epochs, version, classifier, indicesToPredict, modelString, labelFile, numberOfPositions = 9):
	print(f"Training model version {version} for {epochs} epochs.")
	world_size = getMPIWorldSize()
	numberOfModels = 0

	for modelName in models:  
		if modelString and modelString not in modelName:
			continue
		# if modelName == "DQN": 
		if "FullPixelGridML" in modelName:
			lightnModel = TwoPartLightningCNN()
		elif "DQN" in modelName:
			lightnModel = TwoPartLightning(numberOfPositions = numberOfPositions)
		batch_size = 512
		
		lightnDataLoader = ptychographicDataLightning(modelName, classifier = classifier, indicesToPredict = indicesToPredict, labelFile = labelFile, batch_size=batch_size, weighted = False, numberOfPositions = numberOfPositions)
		lightnDataLoader.setup()
		
			
		# if modelName != "DQN":
		# 	model = loadModel(lightnDataLoader.val_dataloader(), modelName)
		# 	lightnModel = lightnModelClass(model)
		early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min", stopping_threshold = 1e-10)
		#swa = StochasticWeightAveraging(swa_lrs=1e-2) #faster but a bit worse (fails, says problem with deepcopy)
		chkpPath = os.path.join("checkpoints",f"{modelName}_{version}")
		if not os.path.exists(chkpPath):
			os.makedirs(chkpPath)
			checkPointExists = False
		else:
			print("loading from checkpoint")
			checkPointExists = True
		checkpoint_callback = ModelCheckpoint(dirpath=chkpPath, save_top_k=1, monitor="val_loss")
		#profiler = AdvancedProfiler(dirpath=".", filename=f"perf_logs_{modelName}_{version}")
		callbacks=[checkpoint_callback]#,early_stop_callback]
		trainer = pl.Trainer(gradient_clip_val=0.5,logger=TensorBoardLogger("tb_logs", log_graph=True,name=f"{modelName}_{version}"),max_epochs=epochs,num_nodes=world_size, accelerator="gpu",devices=1, callbacks=callbacks, log_every_n_steps=1)
		if checkPointExists:
			new_lr = 1e-4
			lightnModel.lr = new_lr
			trainer.fit(lightnModel, datamodule = lightnDataLoader, ckpt_path="last")
		else:
			#Create a Tuner
			tuner = Tuner(trainer)
			# Auto-scale batch size by growing it exponentially
			# if world_size == 1: 
			# 	new_batch_size = tuner.scale_batch_size(lightnModel, datamodule = lightnDataLoader, init_val=512, max_trials= 25) 
			# 	print(f"New batch size: {new_batch_size}")
				# leads to crashing with slurm but has worked with 2048 batch size
				# lightnDataLoader.batch_size is automatically set to new_batch_size
			# finds learning rate automatically
			if world_size == 1:
				# lr_finder = tuner.lr_find(lightnModel, num_training=100, datamodule = lightnDataLoader, min_lr = 1e-11, max_lr = 1 * np.sqrt(lightnDataLoader.batch_size/16), early_stop_threshold=4)
				# assert(lr_finder is not None)
				# new_lr = lr_finder.suggestion()
				new_lr = None
				if (new_lr is None):
					new_lr = 1e-4
				lightnModel.lr = new_lr
			print(f"New learning rate: {lightnModel.lr}")
			trainer.fit(lightnModel, datamodule = lightnDataLoader)
		trainer.save_checkpoint(os.path.join("models",f"{modelName}_{version}.ckpt"))
		if "DQN" in modelName:
			lightnModel = TwoPartLightning().load_from_checkpoint(checkpoint_path = os.path.join("models",f"{modelName}_{version}.ckpt"))
		elif "Zernike" in modelName:
			lightnModel = lightnModelClass(loadModel(lightnDataLoader.val_dataloader(), modelName)).load_from_checkpoint(checkpoint_path = os.path.join("models",f"{modelName}_{version}.ckpt"))
		evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, lightnModel, indicesToPredict, modelName, version, classifier)
		torch.save(lightnModel.state_dict(), os.path.join("models",f"{modelName}_{version}.m"))
		numberOfModels += 1
	print(f"Finished training {numberOfModels} models.")
	if numberOfModels == 0: raise Exception("No model fits the required String.")

if __name__ == '__main__':
	models = ["FullPixelGridML", "ZernikeNormal", "ZernikeBottleneck", "ZernikeComplex", "DQN"]
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--version", type=str, required=True,
	help="version number")
	ap.add_argument("-e", "--epochs", type=int, required=True,
	help="number of epochs")
	ap.add_argument("-m" ,"--models", type=str, required=False, help = "only use models with this String in their name")
	ap.add_argument("-c" ,"--classifier", type=int, required=False, default=0, help = "Use if the model is to be a classfier. Choose the label index to be classified")
	ap.add_argument("-i" ,"--indices", type=str, required=False, default="all", help = "specify indices of labels to predict (eg. '1, 2, 5'). Default is all.")
	ap.add_argument("-np" ,"--numberOfPositions", type=int, required=False, default=9, help = "Specify the number of Positions that the nn gets as input. Default is 9.")
	ap.add_argument("-l" ,"--labelsFile", type=str, required=False, default="labels.csv", help = "Specify the name of the labels-file. Default is labels.csv.")
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
	main(args["epochs"], args["version"], classifier, indicesToPredict, args["models"], args["labelsFile"], numberOfPositions=args["numberOfPositions"])     

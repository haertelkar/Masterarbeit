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
from datasets import ptychographicDataLightning
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment as SLURMEnvironment

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

def loadModel(trainDataLoader, modelName) -> nn.Module:
	# first element to access dimensions
	trainFeatures, trainLabels = next(iter(trainDataLoader))
	numChannels = trainFeatures.shape[1]		

	if modelName == "FullPixelGridML":
		from FullPixelGridML.cnn import cnn
		print("[INFO] initializing the cnn model...")
		model = cnn(
			numChannels=numChannels,
			classes=len(trainLabels[0]))
		
	elif modelName == "unet":
		from FullPixelGridML.unet import unet
		print("[INFO] initializing the unet model...")
		model = unet(
			numChannels=numChannels,
			classes=len(trainLabels[0]))

	elif modelName == "ZernikeBottleneck":
		from Zernike.znnBottleneck import znnBottleneck
		print("[INFO] initializing the znnBottleneck model...")
		model = znnBottleneck(
			inFeatures = len(trainFeatures[0]),
			outFeatures=len(trainLabels[0]))

	elif modelName == "ZernikeNormal":
		from Zernike.znn import znn
		print("[INFO] initializing the znn model...")
		model = znn(
			inFeatures = len(trainFeatures[0]),
			outFeatures=len(trainLabels[0]))
		
	elif modelName == "ZernikeComplex":
		from Zernike.znnMoreComplex import znn
		print("[INFO] initializing the znnComplex model...")
		model = znn(
			inFeatures = len(trainFeatures[0]),
			outFeatures=len(trainLabels[0]))
		
	else:
		raise Exception(f"{modelName} is not a known model")
	
	return model

def evaluater(testDataLoader, test_data, model, indicesToPredict, modelName, version, classifier):
	# we can now evaluate the network on the test set
	print("[INFO] evaluating network...")
	
	# turn off autograd for testing evaluation
	with torch.inference_mode(), open(os.path.join("testDataEval",f'results_{modelName}_{version}.csv'), 'w+', newline='') as resultsTest:
		Writer = csv.writer(resultsTest, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		if not classifier and indicesToPredict is None:
			Writer.writerow(list(test_data.columns[1:])*2)
		elif not classifier:
			Writer.writerow(list(test_data.columns[np.array(indicesToPredict)])*2)
		else:
			Writer.writerow(list(test_data.columns[np.array(indicesToPredict)])) 
		
		model.eval()
		
		# loop over the test set
		for (x, y) in tqdm(testDataLoader, desc="Going through test data"):
			# make the predictions and add them to the list
			pred = model(x)
			if not classifier:
				for predEntry, yEntry in zip(pred.tolist(), y.tolist()):
					predScaled = test_data.scaleUp(row = predEntry)
					yScaled = test_data.scaleUp(row = yEntry)
					Writer.writerow(list(predScaled) +  list(yScaled))
			else:
				for predEntry,yEntry in zip(pred.cpu().numpy(), y.cpu().numpy()):
					Writer.writerow([int(predEntry.argmax() == yEntry.argmax())])
	

def main(epochs, version, classifier, indicesToPredict, modelString):
	world_size = getMPIWorldSize()
	numberOfModels = 0

	for modelName in models:
		if modelString and modelString not in modelName:
			continue
		lightnDataLoader = ptychographicDataLightning(modelName, classifier = classifier, indicesToPredict = indicesToPredict)
		lightnDataLoader.setup()
		model = loadModel(lightnDataLoader.val_dataloader(), modelName)
		lightnModel = lightnModelClass(model)
		early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min", stopping_threshold = 1e-6)
		swa = StochasticWeightAveraging(swa_lrs=1e-2) #faster but a bit less good
		chkpPath = os.path.join("checkpoints",f"{modelName}_{version}")
		if not os.path.exists(chkpPath):
			os.makedirs(chkpPath)
			checkPointExists = False
		else:
			print("loading from checkpoint")
			checkPointExists = True
		trainer = pl.Trainer(plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],logger=TensorBoardLogger("tb_logs", name=f"{modelName}_{version}"),max_epochs=epochs,num_nodes=world_size, accelerator="gpu",devices=1, callbacks=[early_stop_callback, swa], default_root_dir=chkpPath)
		if checkPointExists:
			trainer.fit(lightnModel, datamodule = lightnDataLoader, ckpt_path="last")
		else:
			#Create a Tuner
			tuner = Tuner(trainer)
			# Auto-scale batch size by growing it exponentially
			if world_size == 1: tuner.scale_batch_size(lightnModel, datamodule = lightnDataLoader) 
			# finds learning rate automatically
			tuner.lr_find(lightnModel, datamodule = lightnDataLoader, max_lr = 1e-2, early_stop_threshold = 4)
			trainer.fit(lightnModel, datamodule = lightnDataLoader)
		trainer.save_checkpoint(os.path.join("models",f"{modelName}_{version}.ckpt"))
		lightnModelClass.load_from_checkpoint(checkpoint_path = os.path.join("models",f"{modelName}_{version}.ckpt"), model = model)
		evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, lightnModel, indicesToPredict, modelName, version, classifier)
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
	ap.add_argument("-i" ,"--indices", type=str, required=False, default="all", help = "specify indices of labels to predict (eg. '1, 2, 5'). Default is all.")
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

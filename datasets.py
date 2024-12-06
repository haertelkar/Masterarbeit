import warnings
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import pandas as pd
from rowsIndexToHeader import rowsIndexToHeader
from torch.utils.data import Dataset
import torch
import os
import lightning.pytorch as pl
import h5py
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

class ptychographicDataLightning(pl.LightningDataModule):
	def __init__(self, model_name, batch_size = 2048, num_workers = 20, classifier = False, indicesToPredict = None, labelFile = "labels.csv", testDirectory = "measurements_test", onlyTest = False, weighted  = True):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.model_name = model_name
		# define the train and val splits
		self.TRAIN_SPLIT = 0.75
		self.VAL_SPLIT = 1 - self.TRAIN_SPLIT
		self.classifier = classifier
		self.indicesToPredict = indicesToPredict
		self.setupDone = False
		self.prepare_data_per_node = False
		self.labelFile = labelFile
		self.testDirectory = testDirectory
		self.weights = None
		self.onlyTest = onlyTest
		self.weighted = weighted

	def setup(self, stage = None) -> None:
		if self.setupDone: return
		folderName = None
		if self.model_name == "FullPixelGridML" or self.model_name == "unet":	folderName = "FullPixelGridML"
		elif self.model_name == "ZernikeNormal" or self.model_name == "ZernikeBottleneck" or self.model_name == "ZernikeComplex" or self.model_name == "DQN": folderName = "Zernike"
		else:
			raise Exception(f"model name '{self.model_name}' unknown")
		self.train_dataset = ptychographicData(
					os.path.abspath(os.path.join(folderName,"measurements_train", "train_"+self.labelFile)), 
					os.path.abspath(os.path.join(folderName,"measurements_train")), transform=torch.as_tensor, 
					target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
				)		
		self.val_dataset = ptychographicData(
			os.path.abspath(os.path.join(folderName,"measurements_train", "vali_"+self.labelFile)), 
			os.path.abspath(os.path.join(folderName,"measurements_train")), transform=torch.as_tensor, 
			target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
		)	
		
		self.numTrainSamples = len(self.train_dataset)
		# numValSamples = int(len(self.trainAndVal_dataset) * self.VAL_SPLIT)
		# numValSamples += len(self.trainAndVal_dataset) - self.numTrainSamples - numValSamples
		# (self.train_dataset, self.val_dataset) = random_split(self.trainAndVal_dataset,
		# 	[self.numTrainSamples, numValSamples],
		# 	generator=torch.Generator().manual_seed(42))
		
		self.test_dataset = ptychographicData(
					os.path.abspath(os.path.join(folderName,self.testDirectory, self.labelFile)), 
					os.path.abspath(os.path.join(folderName,self.testDirectory)), transform=torch.as_tensor,
					target_transform=torch.as_tensor, scalingFactors = self.train_dataset.scalingFactors, 
					shift = self.train_dataset.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
				)
		# initialize the train, validation, and test data loaders
		warnings.filterwarnings("ignore", ".*This DataLoader will create 20 worker processes in total. Our suggested max number of worker in current system is 10.*") #this is an incorrect warning as we have 20 cores
		
		xAtomRelPos = None
		yAtomRelPos = None

		if len(self.train_dataset.columns[1:]) == 2:
			if "xAtomRel" in self.train_dataset.columns[1] and "yAtomRel" in self.train_dataset.columns[2]:
				xAtomRelPos = 0
				yAtomRelPos = 1
			elif "xAtomRel" in self.train_dataset.columns[2] and "yAtomRel" in self.train_dataset.columns[1]:
				xAtomRelPos = 1
				yAtomRelPos = 0
		
		if xAtomRelPos is not None and not self.onlyTest and self.weighted:
			self.weights = pd.read_csv(os.path.abspath(os.path.join(folderName,"measurements_train", "weights_"+self.labelFile)), header = None, index_col = None).to_numpy(dtype=float).T[0]
		
		if len(self.train_dataset.columns[1:]) == 1 and not self.onlyTest and self.weighted:
			self.weights = pd.read_csv(os.path.abspath(os.path.join(folderName,"measurements_train", "weights_"+self.labelFile)), header = None, index_col = None).to_numpy(dtype=float).T[0]

		self.setupDone = True

	
	def train_dataloader(self) -> TRAIN_DATALOADERS:
		if self.weights is None: 
			return DataLoader(self.train_dataset,  batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, drop_last=True, shuffle=True)
		else:
			samples_weight = self.weights
			sampler = WeightedRandomSampler(samples_weight, self.numTrainSamples)
			return DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, sampler = sampler, drop_last=True)
	
	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.val_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True)

class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors = None, shift = None, labelIndicesToPredict = None, classifier = False):
		img_labels_pd = pd.read_csv(annotations_file).dropna()
		self.image_labels = img_labels_pd[img_labels_pd.columns[1:]].to_numpy('float32')
		self.image_names = img_labels_pd[img_labels_pd.columns[0]].astype(str)
		self.columns = np.array(list(img_labels_pd.columns))
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.labelIndicesToPredict = labelIndicesToPredict
		self.scalerZernike = 1
		self.classifier = classifier
		self.scalingFactors = None
		self.shift = None
		self.dataPath = os.path.join(self.img_dir, "training_data.hdf5")
		self.dataset = None

		#removed option to classify
		#removed option to select labels to predict
		assert(self.labelIndicesToPredict is None and not self.classifier)

		assert(not(labelIndicesToPredict is 0))
		if isinstance(labelIndicesToPredict, list):
			assert(0 not in labelIndicesToPredict)
		
		# self.createScaleAndShift(scalingFactors = scalingFactors, shift = shift)

		# if self.classifier:
		# 	assert(type(labelIndicesToPredict) != list)
		# 	self.smallestElem = self.img_labels.iloc[:, labelIndicesToPredict].min()
		# 	self.numberOfClasses = self.img_labels.iloc[:, labelIndicesToPredict].max() - self.smallestElem + 1
			

	def __len__(self):
		file_count = len(self.image_labels)
		return file_count

	def __getitem__(self, idx):
		label = self.getLabel(idx)
		# print(self.getImageOrZernike(idx).shape)
		# randCoords = torch.randperm(32*32)[:36]
		# randXCoords = (randCoords % 32)
		# randYCoords = torch.div(randCoords, 32, rounding_mode='floor') 
		randXCoords = torch.tensor([0, 0, 0, 7, 7, 7, 14, 14, 14])
		randYCoords = torch.tensor([0, 7, 14, 0, 7, 14, 0, 7, 14])
		imageOrZernikeMoments = self.getImageOrZernike(idx).reshape((-1,15,15))[:,randXCoords,randYCoords].reshape((9,-1))
		#imageOrZernikeMomentsWithCoords = torch.cat((imageOrZernikeMoments, torch.stack([randXCoords, randYCoords]).T), dim = 1)
		return imageOrZernikeMoments, label
		#print(f"data type: {imageOrZernikeMoments}, data size: {np.shape(imageOrZernikeMoments)}")
		#return imageOrZernikeMoments, label
	
	# def get2Dhist(self):
	# 	xIndex = np.where(self.columns == "xAtomRel")[0][0] - 1
	# 	yIndex = np.where(self.columns == "yAtomRel")[0][0] - 1
	# 	minDist = min(min(self.image_labels[:,xIndex]), min(self.image_labels[:,yIndex]))
	# 	maxDist = max(max(self.image_labels[:,xIndex]), max(self.image_labels[:,yIndex]))
	# 	xBins = np.linspace(minDist, maxDist, 100)
	# 	yBins = np.linspace(minDist, maxDist, 100)
	# 	hist = np.zeros((100,100))
	# 	for rowIndex in range(len(self)):
	# 		xAtomRel = self.image_labels[rowIndex][xIndex]
	# 		yAtomRel = self.image_labels[rowIndex][yIndex]
	# 		xBin = np.digitize(xAtomRel, xBins, right = True)-1
	# 		yBin = np.digitize(yAtomRel, yBins, right = True)-1
	# 		hist[xBin, yBin] += 1
	# 	return hist, xBins, yBins

	def getLabel(self, idx):
		# if self.classifier:
		# 	label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype(int)
		# 	label = self.createOneHotEncodedVector(label)
		# elif self.labelIndicesToPredict is None:
  

		label = self.image_labels[idx] #gets read in as generic objects
		# label = self.scaleDown(label)
		# label = np.linspace(0,1,144).astype('float32') #for testing


		# else:
		# 	label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype('float32') #gets read in as generic objects
		# 	label = self.scaleDown(label)
		if self.target_transform:
			label = self.target_transform(label)
		return np.array(label)

	def getImageOrZernike(self, idx):
		datasetStructIDWithCoords = self.image_names[idx]
		# with h5py.File(self.dataPath,'r') as data:
		# 	imageOrZernikeMoments = np.array(data.get(datasetStructIDWithCoords)).astype('float32')
		if self.dataset is None:
			self.dataset = h5py.File(self.dataPath,'r')
		
		imageOrZernikeMoments = np.array(self.dataset.get(datasetStructIDWithCoords)).astype('float32')	
		if self.transform:
			imageOrZernikeMoments = self.transform(imageOrZernikeMoments) #not scaled here because it has to be datatype uint8 to be scaled automatically
		return imageOrZernikeMoments

	# def createScaleAndShift(self, scalingFactors, shift):
	# 	dontShift = []
	# 	for index in rowsIndexToHeader.keys():
	# 		if "element" or "pixel" in rowsIndexToHeader[index]: #don't scale elements
	# 			dontShift.append(index)

	# 	if self.classifier:
	# 		return
		
	# 	if scalingFactors is not None:
	# 		self.scalingFactors = scalingFactors
	# 	else:
	# 		self.scalingFactors = np.array(self.image_labels.max(axis=0))
	# 		#self.scalingFactors[np.array(dontScale)-1] = 1
	# 		if self.labelIndicesToPredict is not None: self.scalingFactors = self.scalingFactors[np.array(self.labelIndicesToPredict)-1]

	# 	if shift is not None:
	# 		self.shift = shift 
	# 	else:
	# 		self.shift = np.array(self.image_labels.mean(axis=0))
	# 		self.shift[np.array(dontShift)-1] = 0
	# 		if self.labelIndicesToPredict is not None: self.shift = self.shift[np.array(self.labelIndicesToPredict)-1]

	# def createOneHotEncodedVector(self, classIndex):
	# 	classes = np.zeros(self.numberOfClasses)
	# 	classes[classIndex - self.smallestElem] = 1
	# 	return classes

	# def scaleUp(self, row):
	# 	row = np.array(row)
	# 	row *= self.scalingFactors
	# 	row += self.shift
	# 	return row
	
	# def scaleDown(self, row):
	# 	row = np.array(row)
	# 	row -= self.shift
	# 	row /= self.scalingFactors + 0.00000001
	# 	return row

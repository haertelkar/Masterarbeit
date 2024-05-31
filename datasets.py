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
from torch.utils.data import random_split, DataLoader

class ptychographicDataLightning(pl.LightningDataModule):
	def __init__(self, model_name, batch_size = 1024*4, num_workers = 20, classifier = False, indicesToPredict = None):
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

	def setup(self, stage = None) -> None:
		if self.setupDone: return
		folderName = None
		if self.model_name == "FullPixelGridML" or self.model_name == "unet":	folderName = "FullPixelGridML"
		elif self.model_name == "ZernikeNormal" or self.model_name == "ZernikeBottleneck" or self.model_name == "ZernikeComplex": folderName = "Zernike"
		else:
			raise Exception(f"model name '{self.model_name}' unknown")
		self.trainAndVal_dataset = ptychographicData(
					os.path.abspath(os.path.join(folderName,"measurements_train","labels.csv")), 
					os.path.abspath(os.path.join(folderName,"measurements_train")), transform=torch.as_tensor, 
					target_transform=torch.as_tensor, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
				)		
		
		numTrainSamples = int(len(self.trainAndVal_dataset) * self.TRAIN_SPLIT)
		numValSamples = int(len(self.trainAndVal_dataset) * self.VAL_SPLIT)
		numValSamples += len(self.trainAndVal_dataset) - numTrainSamples - numValSamples
		(self.train_dataset, self.val_dataset) = random_split(self.trainAndVal_dataset,
			[numTrainSamples, numValSamples],
			generator=torch.Generator().manual_seed(42))
		
		self.test_dataset = ptychographicData(
					os.path.abspath(os.path.join(folderName,"measurements_test","labels.csv")), 
					os.path.abspath(os.path.join(folderName,"measurements_test")), transform=torch.as_tensor,
					target_transform=torch.as_tensor, scalingFactors = self.trainAndVal_dataset.scalingFactors, 
					shift = self.trainAndVal_dataset.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier
				)
		# initialize the train, validation, and test data loaders
		warnings.filterwarnings("ignore", ".*This DataLoader will create 20 worker processes in total. Our suggested max number of worker in current system is 10.*") #this is an incorrect warning as we have 20 cores
		self.setupDone = True
	
	def train_dataloader(self) -> TRAIN_DATALOADERS:
		return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True)
	
	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.val_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True)

class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors = None, shift = None, labelIndicesToPredict = None, classifier = False):
		img_labels_pd = pd.read_csv(annotations_file)
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
		
		self.createScaleAndShift(scalingFactors = scalingFactors, shift = shift)

		# if self.classifier:
		# 	assert(type(labelIndicesToPredict) != list)
		# 	self.smallestElem = self.img_labels.iloc[:, labelIndicesToPredict].min()
		# 	self.numberOfClasses = self.img_labels.iloc[:, labelIndicesToPredict].max() - self.smallestElem + 1
			

	def __len__(self):
		file_count = len(self.image_labels)
		return file_count

	def __getitem__(self, idx):
		label = self.getLabel(idx)
		imageOrZernikeMoments = self.getImageOrZernike(idx)
		return imageOrZernikeMoments, label
	
	def getLabel(self, idx):
		# if self.classifier:
		# 	label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype(int)
		# 	label = self.createOneHotEncodedVector(label)
		# elif self.labelIndicesToPredict is None:
  

		label = self.image_labels[idx] #gets read in as generic objects
		label = self.scaleDown(label)
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
		# if self.scalerZernike == 1 or len(imageOrZernikeMoments.shape) == 2: #only set once for Zernike ALLWAYS TRUE CURRENTLY SO NOT USED
		# self.scalerZernike = np.max(imageOrZernikeMoments)
		# imageOrZernikeMoments /= self.scalerZernike
		# if len(imageOrZernikeMoments.shape) == 2 and (imageOrZernikeMoments.shape[0]%2 or imageOrZernikeMoments.shape[1]%2) and self.classifier:
		# 	raise Exception("uneven dimension not allowed in unet") #only even dimensions are allowed in unet
		if self.transform:
			imageOrZernikeMoments = self.transform(imageOrZernikeMoments) #not scaled here because it has to be datatype uint8 to be scaled automatically
		return imageOrZernikeMoments

	def createScaleAndShift(self, scalingFactors, shift):
		dontShift = []
		for index in rowsIndexToHeader.keys():
			if "element" or "pixel" in rowsIndexToHeader[index]: #don't scale elements
				dontShift.append(index)

		if self.classifier:
			return
		
		if scalingFactors is not None:
			self.scalingFactors = scalingFactors
		else:
			self.scalingFactors = np.array(self.image_labels.max(axis=0))
			#self.scalingFactors[np.array(dontScale)-1] = 1
			if self.labelIndicesToPredict is not None: self.scalingFactors = self.scalingFactors[np.array(self.labelIndicesToPredict)-1]

		if shift is not None:
			self.shift = shift 
		else:
			self.shift = np.array(self.image_labels.mean(axis=0))
			self.shift[np.array(dontShift)-1] = 0
			if self.labelIndicesToPredict is not None: self.shift = self.shift[np.array(self.labelIndicesToPredict)-1]

	# def createOneHotEncodedVector(self, classIndex):
	# 	classes = np.zeros(self.numberOfClasses)
	# 	classes[classIndex - self.smallestElem] = 1
	# 	return classes

	def scaleUp(self, row):
		row = np.array(row)
		row *= self.scalingFactors
		row += self.shift
		return row
	
	def scaleDown(self, row):
		row = np.array(row)
		row -= self.shift
		row /= self.scalingFactors + 0.00000001
		return row
	
# class multiPos(ptychographicData):
# 	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors=None, shift=None, labelIndicesToPredict=None, classifier=False):
# 		super().__init__(annotations_file, img_dir, transform, target_transform, scalingFactors, shift, labelIndicesToPredict, classifier)
# 		self.scansPerSpecimen = 100
# 		self.numberOfSpecimen = None
# 		self.indicesPerSpecimen = 20
# 		self.idxToXY = {}
# 		self.indices = self.getIndicesOfSparseScan(indicesPerSpecimen = self.indicesPerSpecimen, outputsPerSpecimen = self.scansPerSpecimen)

# 	def getIndicesOfSparseScan(self, indicesPerSpecimen, outputsPerSpecimen):
# 		allFileNames = self.img_labels.iloc[:, 0]
# 		specimenToIndex = {}
# 		self.numberOfSpecimen = 0
# 		for idx, fileName in enumerate(allFileNames):
# 			_, specimen, xPos, yPos, _, _, _ = fileName.split("_")
# 			self.idxToXY[idx] = (float(xPos), float(yPos))
# 			if specimen not in specimenToIndex: 
# 				specimenToIndex[specimen] = [idx]
# 				self.numberOfSpecimen += 1
# 			else:
# 				specimenToIndex[specimen].append(idx)

# 		randomIdxArray = []
# 		for idxArray in specimenToIndex.values():
# 			for _ in range(outputsPerSpecimen): 
# 				randomIdxArray.append(np.random.choice(idxArray, size = indicesPerSpecimen, replace= False))
		
# 		return np.array(randomIdxArray)

# 	def __len__(self):
# 		return self.scansPerSpecimen * self.numberOfSpecimen

# 	def __getitem__(self, sample):
# 		zernikeMultiPos = None
# 		labelMultiPos = None
# 		# XYs = []


# 		for idx in self.indices[sample]:
# 			if labelMultiPos is None: labelMultiPos = self.getLabel(idx)
# 			if zernikeMultiPos is None: zernikeMultiPos = self.getImageOrZernike(idx)

# 			x,y = self.idxToXY.get(idx)
# 			labelMultiPos = np.append(labelMultiPos,self.getLabel(idx))
# 			zernikeMultiPos = torch.cat((torch.Tensor((x,y)), zernikeMultiPos,self.getImageOrZernike(idx)))
# 			# XYs += [x,y] 



# 		# zernikeMultiPos = np.append(np.array(zernikeMultiPos).flatten(),XYs) #TODO only valid for znn, wont work with cnn, for cnn use layers (also no idea how to implement xy for cnn)
# 		# labelMultiPos = np.array(labelMultiPos).flatten()
# 		#TODO missing Xpos And Ypos Also Labels Should Be Cleaned Up


# 		#TODO ideally one label with the absolute pos of all atoms and also not a random SparseScan
# 		#TODO easier: four close positions -> probably needs different simulateMoreComplex.py
# 		return zernikeMultiPos, labelMultiPos
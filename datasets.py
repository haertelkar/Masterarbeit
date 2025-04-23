from random import randint
import warnings
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import pandas as pd
from rowsIndexToHeader import rowsIndexToHeader
from torch.utils.data import Dataset
import torch
import os
import lightning as L
import h5py
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Separate data and labels

    # Pad sequences to the longest length in batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Create attention mask (True for padded positions)
    mask = (padded_sequences.abs().sum(dim=2) == 0)

    return padded_sequences, torch.stack(labels), mask

def collate_fn_gru(batch):
	sequences, labels = zip(*batch)  # Separate data and labels
	sequence_lengths = torch.tensor([len(seq) for seq in sequences])
    # Pad sequences to the longest length in batch
	padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
	pack_padded_sequence(padded_sequences.unsqueeze(-1).float(), 
                                    sequence_lengths, batch_first=True, enforce_sorted=False)
	return padded_sequences, torch.stack(labels), torch.zeros((1,1))



colFun = collate_fn

class ptychographicDataLightning(L.LightningDataModule):
	def __init__(self, model_name, batch_size = 1024, num_workers = 20, classifier = False, indicesToPredict = None, labelFile = "labels.csv", testDirectory = "measurements_test", onlyTest = False, weighted  = True, numberOfPositions = 9):
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
		self.numberOfPositions = numberOfPositions

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
					target_transform=None,#torch.as_tensor,
					labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, numberOfPositions = self.numberOfPositions
				)		
		self.val_dataset = ptychographicData(
			os.path.abspath(os.path.join(folderName,"measurements_train", "vali_"+self.labelFile)), 
			os.path.abspath(os.path.join(folderName,"measurements_train")), transform=torch.as_tensor, 
			target_transform=None,#torch.as_tensor, 
			labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, numberOfPositions = self.numberOfPositions
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
					target_transform=None,#torch.as_tensor,
					shift = self.train_dataset.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, numberOfPositions = self.numberOfPositions
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
			return DataLoader(self.train_dataset,  batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, drop_last=True, shuffle=True, collate_fn=colFun)
		else:
			samples_weight = self.weights
			sampler = WeightedRandomSampler(samples_weight, self.numTrainSamples)
			return DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, sampler = sampler, drop_last=True, collate_fn=colFun)
	
	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.val_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True, collate_fn=colFun)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True, collate_fn=colFun)

class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, shift = None, labelIndicesToPredict = None, classifier = False, numberOfPositions = 9):
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
		with open('Zernike/stdValues.csv', 'r') as file:
			line = file.readline()
			self.stdValues = [float(x.strip()) for x in line.split(',') if x.strip()]
			self.stdValuesArray = np.array(self.stdValues[:-3] + [1,1,1])
		with open('Zernike/meanValues.csv', 'r') as file:
			line = file.readline()
			self.meanValues = [float(x.strip()) for x in line.split(',') if x.strip()]
			self.meanValuesArray : np.ndarray = np.array(self.meanValues[:-3] + [0,0,0])
		self.shift = None
		self.dataPath = os.path.join(self.img_dir, f"training_data.hdf5")
		self.dataset = None
		self.numberOfPositions = numberOfPositions
		self.numberOfZernikeMoments  = 40
		self.zernikeLength = self.zernikeLengthCalc()
		print(f"Loading {self.dataPath} with {len(self.image_labels)} zernike data. The data is using {self.numberOfZernikeMoments} zernike moments (length = {self.zernikeLength}) and a maximum of {self.numberOfPositions} positions.")

		#removed option to classify
		#removed option to select labels to predict
		assert(self.labelIndicesToPredict is None and not self.classifier)

		assert(labelIndicesToPredict != 0)
		if isinstance(labelIndicesToPredict, list):
			assert(0 not in labelIndicesToPredict)
		
		# self.createScaleAndShift(scalingFactors = scalingFactors, shift = shift)

		# if self.classifier:
		# 	assert(type(labelIndicesToPredict) != list)
		# 	self.smallestElem = self.img_labels.iloc[:, labelIndicesToPredict].min()
		# 	self.numberOfClasses = self.img_labels.iloc[:, labelIndicesToPredict].max() - self.smallestElem + 1
	def zernikeLengthCalc(self):
		resultVectorLength = 0 
		numberOfOSAANSIMoments = self.numberOfZernikeMoments
		for n in range(numberOfOSAANSIMoments + 1):
			for mShifted in range(2*n+1):
				m = mShifted - n
				if (n-m)%2 != 0:
					continue
				resultVectorLength += 1
		return resultVectorLength		

	def __len__(self):
		file_count = len(self.image_labels)
		return file_count

	def __getitem__(self, idx):
		label = self.getLabel(idx)
		imageOrZernikeMomentsWithCoordsAndPad = self.getImageOrZernike(idx)
		# print(imageOrZernikeMomentsWithCoordsAndPad)
		# length = min(len(imageOrZernikeMomentsWithCoordsAndPad), 100)
		return imageOrZernikeMomentsWithCoordsAndPad , label
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
		return torch.as_tensor(np.array(label))

	def getImageOrZernike(self, idx):
		datasetStructID = self.image_names[idx]
		# with h5py.File(self.dataPath,'r') as data:
		# 	imageOrZernikeMoments = np.array(data.get(datasetStructIDWithCoords)).astype('float32')
		if self.dataset is None:
			self.dataset = h5py.File(self.dataPath,'r')
		try:
			imageOrZernikeMoments = np.array(self.dataset[datasetStructID])	
		except KeyError:
			imageOrZernikeMoments = np.array(self.dataset["0" + datasetStructID]) #sometimes the leading zero goes missing
		except OSError as e:
			print(e)
			raise Exception(f"OSError: {datasetStructID} in {self.dataPath}")
		imageOrZernikeMoments = (imageOrZernikeMoments - self.meanValuesArray[np.newaxis,:]) / self.stdValuesArray[np.newaxis,:]
		imageOrZernikeMoments = imageOrZernikeMoments.astype('float32')
		if self.numberOfZernikeMoments != 40:
			imageOrZernikeMoments = np.delete(imageOrZernikeMoments, np.s_[self.zernikeLength + 1,-3], axis=1) #remove the higher order zernike moments but keep x,y and padding
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

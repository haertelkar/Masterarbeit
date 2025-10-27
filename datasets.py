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
    sequences , labels = zip(*batch)  # Separate data and labels

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

#def collate_fn(batch):
 #   sequences, labels = zip(*batch)
#
 #   max_len = max(len(seq) for seq in sequences)
#    B = len(sequences)
 #   H, W = sequences[0][0].shape  # [H, W]
#
 #   padded_sequences = torch.zeros((B, max_len, 1, H, W))
#    mask = torch.zeros((B, max_len), dtype=torch.bool)
#
 #   for i, seq in enumerate(sequences):
#        seq_len = len(seq)
 #       padded_sequences[i, :seq_len] = torch.stack(seq)  # [seq_len, 1, H, W]
#        mask[i, :seq_len] = True
#
 #   return padded_sequences, torch.stack(labels), mask



colFun = collate_fn

class ptychographicDataLightning(L.LightningDataModule):
	def __init__(self, model_name, batch_size = 128, num_workers = 20, classifier = False, indicesToPredict = None, 
			    labelFile = "labels.csv", trainDirectories = ["measurements_train"],testDirectories = ["measurements_test"],
			    onlyTest = False, weighted  = True, numberOfPositions = 20, numberOfZernikeMoments = 860, lessBorder = 15, sparsity = 1, number_of_samples = 0):
		super().__init__()
		print(f"Using num workers {num_workers}")
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
		self.testDirectories = testDirectories
		self.weights = None
		self.onlyTest = onlyTest
		self.weighted = weighted
		self.numberOfPositions = numberOfPositions
		self.numberOfZernikeMoments = numberOfZernikeMoments
		self.trainDirectories = trainDirectories
		self.testDirectories = testDirectories
		self.lessBorder = lessBorder
		self.sparsity = sparsity
		self.number_of_samples = number_of_samples

	def setup(self, stage = None) -> None:
		if self.setupDone: return
		folderName = None
		if self.model_name == "FullPixelGridML" or self.model_name == "unet" or self.model_name == "cnnTransformer" or self.model_name == "visionTransformer":	folderName = "FullPixelGridML"
		elif self.model_name == "ZernikeNormal" or self.model_name == "ZernikeBottleneck" or self.model_name == "ZernikeComplex" or self.model_name == "TrE": folderName = "Zernike"
		else:
			raise Exception(f"model name '{self.model_name}' unknown")
		trainDatasets = []
		valiDatasets = []
		testDatasets = []

		for cnt, (testDirectory, trainDirectory) in enumerate(zip(self.testDirectories, self.trainDirectories)):
			train_dataset =  ptychographicData(
						os.path.abspath(os.path.join(folderName,trainDirectory, "train_"+self.labelFile)), 
						os.path.abspath(os.path.join(folderName,trainDirectory)), transform=torch.as_tensor, 
						target_transform=None,#torch.as_tensor,
						labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, 
						numberOfPositions = self.numberOfPositions, numberOfZernikeMoments = self.numberOfZernikeMoments,
						lessBorder = self.lessBorder, sparsity = self.sparsity, number_of_samples = self.number_of_samples
					)		
			val_dataset = ptychographicData(
				os.path.abspath(os.path.join(folderName,trainDirectory, "vali_"+self.labelFile)), 
				os.path.abspath(os.path.join(folderName,trainDirectory)), transform=torch.as_tensor, 
				target_transform=None,#torch.as_tensor, 
				labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, numberOfPositions = self.numberOfPositions, 
				numberOfZernikeMoments = self.numberOfZernikeMoments, lessBorder = self.lessBorder, sparsity = self.sparsity
			)	
			
			self.numTrainSamples = len(train_dataset)
			# numValSamples = int(len(self.trainAndVal_dataset) * self.VAL_SPLIT)
			# numValSamples += len(self.trainAndVal_dataset) - self.numTrainSamples - numValSamples
			# (self.train_dataset, self.val_dataset) = random_split(self.trainAndVal_dataset,
			# 	[self.numTrainSamples, numValSamples],
			# 	generator=torch.Generator().manual_seed(42))
			
			test_dataset = ptychographicData(
						os.path.abspath(os.path.join(folderName,testDirectory, self.labelFile)), 
						os.path.abspath(os.path.join(folderName,testDirectory)), transform=torch.as_tensor,
						target_transform=None,#torch.as_tensor,
						shift = train_dataset.shift, labelIndicesToPredict= self.indicesToPredict, classifier= self.classifier, 
						numberOfPositions = self.numberOfPositions, numberOfZernikeMoments = self.numberOfZernikeMoments, lessBorder = self.lessBorder, sparsity = self.sparsity
					)
			# initialize the train, validation, and test data loaders
			warnings.filterwarnings("ignore", ".*This DataLoader will create 20 worker processes in total. Our suggested max number of worker in current system is 10.*") #this is an incorrect warning as we have 20 cores
			
			# xAtomRelPos = None
			# yAtomRelPos = None

			# if len(train_dataset.columns[1:]) == 2:
			# 	if "xAtomRel" in train_dataset.columns[1] and "yAtomRel" in train_dataset.columns[2]:
			# 		xAtomRelPos = 0
			# 		yAtomRelPos = 1
			# 	elif "xAtomRel" in train_dataset.columns[2] and "yAtomRel" in train_dataset.columns[1]:
			# 		xAtomRelPos = 1
			# 		yAtomRelPos = 0
			
			# if xAtomRelPos is not None and not self.onlyTest and self.weighted:
			# 	self.weights = pd.read_csv(os.path.abspath(os.path.join(folderName,self.trainDirectories, "weights_"+self.labelFile)), header = None, index_col = None).to_numpy(dtype=float).T[0]
			
			# if len(train_dataset.columns[1:]) == 1 and not self.onlyTest and self.weighted:
			# 	self.weights = pd.read_csv(os.path.abspath(os.path.join(folderName,self.trainDirectories, "weights_"+self.labelFile)), header = None, index_col = None).to_numpy(dtype=float).T[0]
			trainDatasets.append(train_dataset)
			valiDatasets.append(val_dataset)
			# testDatasets.append(test_dataset)
		self.train_dataset = torch.utils.data.ConcatDataset(trainDatasets)
		self.val_dataset = torch.utils.data.ConcatDataset(valiDatasets)
		self.test_dataset = test_dataset#torch.utils.data.ConcatDataset(testDatasets)
		self.setupDone = True

	
	def train_dataloader(self) -> TRAIN_DATALOADERS:
		# if self.weights is None: 
		return DataLoader(self.train_dataset,  batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, drop_last=True, shuffle=True, collate_fn=colFun)
		# else:
		# 	samples_weight = self.weights
		# 	sampler = WeightedRandomSampler(samples_weight, self.numTrainSamples)
		# 	return DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers= self.num_workers, pin_memory=True, sampler = sampler, drop_last=True, collate_fn=colFun)
	
	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.val_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True, collate_fn=colFun)

	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= self.num_workers, pin_memory=True, collate_fn=colFun)

class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, shift = None, labelIndicesToPredict = None, 
			  classifier = False, numberOfPositions = 9, numberOfZernikeMoments = 860, lessBorder = 15, sparsity = 1, number_of_samples = 0):
		self.lessBorder = lessBorder
		img_labels_pd = pd.read_csv(annotations_file).dropna()
		if number_of_samples != 0:
			number_of_samples = min((len(img_labels_pd), number_of_samples))
			print(f"limited data to {number_of_samples} samples of {len(img_labels_pd)}")
			img_labels_pd = img_labels_pd[:number_of_samples]
		self.image_labels = img_labels_pd[img_labels_pd.columns[1:]].to_numpy('float32')
		self.image_names = img_labels_pd[img_labels_pd.columns[0]].astype(str)
		self.columns = np.array(list(img_labels_pd.columns))
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.labelIndicesToPredict = labelIndicesToPredict
		self.classifier = classifier
		self.zernike = True if "Zernike" in self.img_dir else False
		if self.zernike:
			self.actualLenZernike = 0
			self.numberOfZernikeMoments  = numberOfZernikeMoments
			with open('Zernike/stdValues.csv', 'r') as file:
				line = file.readline()
				self.stdValues = [float(x.strip()) for x in line.split(',') if x.strip()]
				self.stdValuesArray = np.array(self.stdValues) #TODO previously there was 1,1 as scaling for the coordinates
				self.stdValuesArray = np.delete(self.stdValuesArray, np.s_[self.numberOfZernikeMoments:-2])
				self.stdValuesArray = np.clip(self.stdValuesArray, a_min=1e-5, a_max=None) #avoid division by zero
			with open('Zernike/meanValues.csv', 'r') as file:
				line = file.readline()
				self.meanValues = [float(x.strip()) for x in line.split(',') if x.strip()]
				self.meanValuesArray : np.ndarray = np.array(self.meanValues)  #TODO previously there was 0,0 as mean for the coordinates
				self.meanValuesArray = np.delete(self.meanValuesArray, np.s_[self.numberOfZernikeMoments:-2])
		self.shift = None
		self.dataPath = os.path.join(self.img_dir, f"training_data.hdf5")
		self.dataset = None
		self.numberOfPositions = numberOfPositions
		self.sparsity = sparsity
		
		

		
		if self.zernike: print(f"Loading {self.dataPath} with {len(self.image_labels)} zernike data. The data is using {self.numberOfZernikeMoments} zernike moments and a maximum of {self.numberOfPositions} positions.")

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
			try:	
				imageOrZernikeMoments = np.array(self.dataset["0" + datasetStructID]) #sometimes the leading zero goes missing
			except KeyError:
				imageOrZernikeMoments = np.array(self.dataset[datasetStructID])
		except OSError as e:
			print(e)
			raise Exception(f"OSError: {datasetStructID} in {self.dataPath}")
		if self.zernike:
			if self.actualLenZernike == 0:
				self.actualLenZernike = imageOrZernikeMoments.shape[1] - 2
			if self.lessBorder < 15:
				BegrenzungAussen = self.lessBorder #less from the outer border
				
				imageOrZernikeMoments = imageOrZernikeMoments[(imageOrZernikeMoments[:,-2] >  -BegrenzungAussen) & (imageOrZernikeMoments[:,-2] < (BegrenzungAussen + 15))]
				imageOrZernikeMoments = imageOrZernikeMoments[(imageOrZernikeMoments[:,-1] >  -BegrenzungAussen) & (imageOrZernikeMoments[:,-1] < (BegrenzungAussen + 15))]
			if self.numberOfZernikeMoments != 860 and self.actualLenZernike != self.numberOfZernikeMoments:
				imageOrZernikeMoments = np.delete(imageOrZernikeMoments, np.s_[self.numberOfZernikeMoments:-2], axis=1) #remove the higher order zernike moments but keep x,y
			if self.sparsity > 1:
				xOffset = imageOrZernikeMoments[0,-2] % self.sparsity
				yOffset = imageOrZernikeMoments[0,-1] % self.sparsity
				xEverySthCordinate = (imageOrZernikeMoments[:,-2] % self.sparsity) == xOffset
				yEverySthCordinate = (imageOrZernikeMoments[:,-1] % self.sparsity) == yOffset
				imageOrZernikeMoments = imageOrZernikeMoments[xEverySthCordinate & yEverySthCordinate]
			imageOrZernikeMoments = (imageOrZernikeMoments - self.meanValuesArray[np.newaxis,:]) / self.stdValuesArray[np.newaxis,:]
		

		imageOrZernikeMoments = imageOrZernikeMoments.astype('float32')

		if self.transform:
			imageOrZernikeMoments = self.transform(imageOrZernikeMoments) #not scaled here because it has to be datatype uint8 to be scaled automatically
		return imageOrZernikeMoments

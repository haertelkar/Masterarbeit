import numpy as np
import pandas as pd
from rowsIndexToHeader import rowsIndexToHeader
from torch.utils.data import Dataset
import torch

class ptychographicData(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors = None, shift = None, labelIndicesToPredict = None, classifier = False):
		self.img_labels = pd.read_csv(annotations_file)
		self.columns = np.array(list(self.img_labels.columns))
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform
		self.labelIndicesToPredict = labelIndicesToPredict
		self.scalerZernike = 1
		self.classifier = classifier
		self.scalingFactors = None
		self.shift = None

		assert(not(labelIndicesToPredict is 0))
		if isinstance(labelIndicesToPredict, list):
			assert(0 not in labelIndicesToPredict)
		
		self.createScaleAndShift(scalingFactors = scalingFactors, shift = shift)

		if self.classifier:
			assert(type(labelIndicesToPredict) != list)
			self.smallestElem = self.img_labels.iloc[:, labelIndicesToPredict].min()
			self.numberOfClasses = self.img_labels.iloc[:, labelIndicesToPredict].max() - self.smallestElem + 1
			

	def __len__(self):
		file_count = len(self.img_labels)
		return file_count

	def __getitem__(self, idx):
		label = self.getLabel(idx)
		imageOrZernikeMoments = self.getImageOrZernike(idx)
		return imageOrZernikeMoments, label
	
	def getLabel(self, idx):
		if self.classifier:
			label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype(int)
			label = self.createOneHotEncodedVector(label)
		elif self.labelIndicesToPredict is None:
			label = np.array(self.img_labels.iloc[idx, 1:]).astype('float32') #gets read in as generic objects
			label = self.scaleDown(label)
		else:
			label = np.array(self.img_labels.iloc[idx, self.labelIndicesToPredict]).astype('float32') #gets read in as generic objects
			label = self.scaleDown(label)
		if self.target_transform:
			label = self.target_transform(label)
		return np.array(label)

	def getImageOrZernike(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		imageOrZernikeMoments = np.load(img_path).astype('float32')		
		if self.scalerZernike == 1 or len(imageOrZernikeMoments.shape) == 2: #only set once for Zernike
			self.scalerZernike = np.max(imageOrZernikeMoments)
		imageOrZernikeMoments /= self.scalerZernike
		if len(imageOrZernikeMoments.shape) == 2 and (imageOrZernikeMoments.shape[0]%2 or imageOrZernikeMoments.shape[1]%2) and self.classifier:
			raise Exception("uneven dimension not allowed in unet") #only even dimensions are allowed in unet
		if self.transform:
			imageOrZernikeMoments = self.transform(imageOrZernikeMoments) #not scaled here because it has to be datatype uint8 to be scaled automatically
		return imageOrZernikeMoments

	def createScaleAndShift(self, scalingFactors, shift):
		dontShift = []
		for index in rowsIndexToHeader.keys():
			if "element" in rowsIndexToHeader[index]: #don't scale elements
				dontShift.append(index)

		if self.classifier:
			return
		
		if scalingFactors is not None:
			self.scalingFactors = scalingFactors
		else:
			self.scalingFactors = np.array(self.img_labels.iloc[:, 1:].max(axis=0).to_list())
			#self.scalingFactors[np.array(dontScale)-1] = 1
			if self.labelIndicesToPredict is not None: self.scalingFactors = self.scalingFactors[np.array(self.labelIndicesToPredict)-1]

		if shift is not None:
			self.shift = shift 
		else:
			self.shift = np.array(self.img_labels.iloc[:, 1:].mean(axis=0).to_list())
			self.shift[np.array(dontShift)-1] = 0
			if self.labelIndicesToPredict is not None: self.shift = self.shift[np.array(self.labelIndicesToPredict)-1]

	def createOneHotEncodedVector(self, classIndex):
		classes = np.zeros(self.numberOfClasses)
		classes[classIndex - self.smallestElem] = 1
		return classes

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
	
class multiPos(ptychographicData):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, scalingFactors=None, shift=None, labelIndicesToPredict=None, classifier=False):
		super().__init__(annotations_file, img_dir, transform, target_transform, scalingFactors, shift, labelIndicesToPredict, classifier)
		self.scansPerSpecimen = 100
		self.numberOfSpecimen = None
		self.indicesPerSpecimen = 20
		self.idxToXY = {}
		self.indices = self.getIndicesOfSparseScan(indicesPerSpecimen = self.indicesPerSpecimen, outputsPerSpecimen = self.scansPerSpecimen)

	def getIndicesOfSparseScan(self, indicesPerSpecimen, outputsPerSpecimen):
		allFileNames = self.img_labels.iloc[:, 0]
		specimenToIndex = {}
		self.numberOfSpecimen = 0
		for idx, fileName in enumerate(allFileNames):
			_, specimen, xPos, yPos, _, _, _ = fileName.split("_")
			self.idxToXY[idx] = (float(xPos), float(yPos))
			if specimen not in specimenToIndex: 
				specimenToIndex[specimen] = [idx]
				self.numberOfSpecimen += 1
			else:
				specimenToIndex[specimen].append(idx)

		randomIdxArray = []
		for idxArray in specimenToIndex.values():
			for _ in range(outputsPerSpecimen): 
				randomIdxArray.append(np.random.choice(idxArray, size = indicesPerSpecimen, replace= False))
		
		return np.array(randomIdxArray)

	def __len__(self):
		return self.scansPerSpecimen * self.numberOfSpecimen

	def __getitem__(self, sample):
		zernikeMultiPos = None
		labelMultiPos = None
		# XYs = []


		for idx in self.indices[sample]:
			if labelMultiPos is None: labelMultiPos = self.getLabel(idx)
			if zernikeMultiPos is None: zernikeMultiPos = self.getImageOrZernike(idx)

			x,y = self.idxToXY.get(idx)
			labelMultiPos = np.append(labelMultiPos,self.getLabel(idx))
			zernikeMultiPos = torch.cat((torch.Tensor((x,y)), zernikeMultiPos,self.getImageOrZernike(idx)))
			# XYs += [x,y] 



		# zernikeMultiPos = np.append(np.array(zernikeMultiPos).flatten(),XYs) #TODO only valid for znn, wont work with cnn, for cnn use layers (also no idea how to implement xy for cnn)
		# labelMultiPos = np.array(labelMultiPos).flatten()
		#TODO missing Xpos And Ypos Also Labels Should Be Cleaned Up


		#TODO ideally one label with the absolute pos of all atoms and also not a random SparseScan
		#TODO easier: four close positions -> probably needs different simulateMoreComplex.py
		return zernikeMultiPos, labelMultiPos
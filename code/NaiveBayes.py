import pandas as pd 
import numpy as np 
import math
from heapq import nsmallest
import scipy.stats as st
from scipy.stats import norm



def calculateStats(groundTruth, prediction):#testset is the groundtruth and knn_final is the prediction
	mTP=0
	mTN=0
	mFP=0
	mFN=0
	for i in range(0, len(groundTruth)):
		if groundTruth[i]== 1 and prediction[i]==1:
			mTP+=1
		elif groundTruth[i]== 0 and prediction[i]==0:
			mTN+=1
		elif groundTruth[i]== 0 and prediction[i]==1:
			mFP+=1
		elif groundTruth[i]== 1 and prediction[i]==0:
			mFN+=1
	print("#################")
	print(mTP,mTN,mFP,mFN)
	print("################")
	return mTP,mTN,mFP,mFN

#calculateStats(testSet[:,-1],knn_final )
def getAccuracy(mTP,mTN,mFP,mFN):
	return (mTP+mTN)/ (mTP+mTN+mFP+mFN)

def getRecall(mTP,mTN,mFP,mFN):
	if (mTP + mFP) == 0:
		return 0
	return mTP / (mTP + mFN)

def getPrecision(mTP,mTN,mFP,mFN):
	if (mTP + mFP) == 0:
		return 0
	return mTP / (mTP + mFP)

def getF_1Measure(mTP,mTN,mFP,mFN):
	precision = getPrecision(mTP,mTN,mFP,mFN);
	recall = getRecall(mTP,mTN,mFP,mFN);
	if (precision + recall) == 0:
		return 0
	return ((2 * precision * recall) / (precision + recall));

def calculateProbablities(arr):
	p_map = {}
	arr_len = len(arr)
	for item in arr:
		if item not in p_map:
			p_map[item] = 0.0
		p_map[item] += 1.0

	for key in p_map:
		p_map[key] = p_map[key] / arr_len
	return p_map

def findProbablity(prob_map, key):
	if key in prob_map:
		return prob_map[key]
	return 0

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def calculateMeanSd(arr):
	p_map = {}
	p_map['mean'] = mean(arr)
	p_map['sd'] = stdev(arr)
	return p_map

def calculateNormalProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def DataSet(filename,split):
	predicted_labels = []
	data_2= pd.read_csv(filename, header=None, delimiter= "\t")
	df= pd.DataFrame(data_2)
	number= df._get_numeric_data().columns

	#array_data converts to a matrix
	array_data= df.as_matrix(columns= None)#None means return all the columns

	data_float=[]
	data_matrix=[]

	##########converting all values to float
	for i in range(0, len(array_data)):
		for j in range(0, len(array_data[i])-1):
			if((j in number)):# here j and column number in "number" match
				array_data[i,j]= float(array_data[i,j])
	
	cut = int(0.9*len(array_data))#trainset
	trainSet = array_data[0:cut,:]
	testSet = array_data[cut:len(array_data),:]
	trainSet_Zero=[]
	trainSet_One= []
	for row in range(0, len(trainSet)):
		if trainSet[row, -1] == 0:
			trainSet_Zero.append(trainSet[row])
		if trainSet[row, -1] == 1:
			trainSet_One.append(trainSet[row])

	trainSet_Zero= np.array(trainSet_Zero)
	trainSet_One= np.array(trainSet_One)
	

	ph_zero = len(trainSet_Zero)/len(trainSet)
	ph_one = len(trainSet_One)/len(trainSet)
	p_X_H_zero = []
	p_X_H_one = []
	p_X_all = []
	for i in range(0, len(trainSet[0]) - 1):
		if type(trainSet[0, i]) == str:
			p_X_H_zero.append(calculateProbablities(trainSet_Zero[:, i]))
			p_X_H_one.append(calculateProbablities(trainSet_One[:, i]))
			p_X_all.append(calculateProbablities(trainSet[:, i]))
		else:
			p_X_H_zero.append(calculateMeanSd(trainSet_Zero[:, i]))
			p_X_H_one.append(calculateMeanSd(trainSet_One[:, i]))
			p_X_all.append(calculateMeanSd(trainSet[:, i]))

	
	for i in range(0, len(testSet)):
		p_X_H_z = 1
		p_X_H_o = 1
		p_X = 1
		for j in range(0, len(testSet[i]) - 1):
			if type(testSet[i, j]) == str:
				p_map_z = p_X_H_zero[j]
				p_X_H_z = p_X_H_z * findProbablity(p_map_z, testSet[i,j])
				p_map_o = p_X_H_one[j]
				p_X_H_o *= findProbablity(p_map_o, testSet[i,j])
				p_map_all = p_X_all[j]
				p_X *= findProbablity(p_map_all, testSet[i,j])
			else:
				p_map_z = p_X_H_zero[j]
				p_X_H_z = p_X_H_z * calculateNormalProbability(testSet[i,j], p_map_z['mean'], p_map_z['sd'])
				p_map_o = p_X_H_one[j]
				p_X_H_o *= calculateNormalProbability(testSet[i,j], p_map_o['mean'], p_map_o['sd'])
				p_map_all = p_X_all[j]
				p_X *= calculateNormalProbability(testSet[i,j], p_map_all['mean'], p_map_all['sd'])
		p_H_o_X = (p_X_H_o * ph_one) / (p_X + 0.000000000000001)
		p_H_z_X = (p_X_H_z * ph_zero) / (p_X + 0.00000000000001)
		print ("p(H0|X)=" + str(p_H_z_X) + "  p(H1|X)=" + str(p_H_o_X))
		if p_H_o_X > p_H_z_X:
			predicted_labels.append(1.0)
		else:
			predicted_labels.append(0.0)
	print(testSet)
	mTP,mTN,mFP,mFN=calculateStats(testSet[:,-1], predicted_labels)
	return getAccuracy(mTP,mTN,mFP,mFN),getPrecision(mTP,mTN,mFP,mFN), getRecall(mTP,mTN,mFP,mFN), getF_1Measure(mTP,mTN,mFP,mFN)



print(DataSet("project3_dataset2.txt", .1))#split can be changes here


Acc_1=0
Prec_1=0
Rec_1=0
F1_1 =0
for i in range(0,10):
	Acc,Prec, Rec, F1=DataSet("project3_dataset2.txt", .1*i)

	Acc_1+= Acc
	Prec_1+= Prec
	Rec_1 += Rec
	F1_1 += F1
print("FInalllllllllll")
print((Acc_1/10)*100," ", (Prec_1/10*100)," ",(Rec_1/10*100), " ", (F1_1/10*100))





















import pandas as pd 
import numpy as np 
import math
from heapq import nsmallest


def euclidean(a, b):
	distance = 0
	z= 0
	for i in range(0, len(a)-1):
		if((type(b[i]))== str or (type(a[i]))== str):# this if condition is for categoricl variables
			z+=1
			if(b[i]==a[i]):
				b[i]=0
				a[i]=0
				#print("changed to 0")
			else:
				b[i]=0
				a[i]=1
				#print("changed to 1")
		
		distance += ((b[i] - a[i])) ** 2
		#print(distance)
	final_euc= math.sqrt(distance)
	#print("euc:",final_euc)
	return final_euc

def normalization(train):# ':'' means everything and 0 is first row
#running through columns
	for i in range(0,len(train[0])-1):
		if(type(train[0,i]) != str):
			mean= np.mean((train[:,i]))
			stdev= np.std((train[:,i]))
			train[:,i] = ((train[:,i] - mean) / stdev)
	return train

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
	return mTP / (mTP + mFN)
def getPrecision(mTP,mTN,mFP,mFN):
	return mTP / (mTP + mFP)
def getF_1Measure(mTP,mTN,mFP,mFN):
	precision = getPrecision(mTP,mTN,mFP,mFN);
	recall = getRecall(mTP,mTN,mFP,mFN);
	return ((2 * precision * recall) / (precision + recall));



def DataSet(filename, split):
	
	data_2= pd.read_csv(filename, header=None, delimiter= "\t")


	df= pd.DataFrame(data_2)
	number= df._get_numeric_data().columns
	#print("Numeric columns", number)
	#array_data converts to a matrix
	array_data= df.as_matrix(columns= None)#None means return all the columns
	##########converting all values to float, 
	#####only values with column number similar to "number" variable above that has only numeric column number in it

	for i in range(0, len(array_data)):
		
		for j in range(0, len(array_data[i])-1):
			#print("number",number)
			if((j in number)):# here j and column number in "number" match
				#print("equal j and number", j, number[j])
				array_data[i,j]= float(array_data[i,j])
	
	array_data=normalization(array_data)
	########SPLITING THE DATA
	#TestRow is an array with values 200:300 when split=0.2 given there are 1000 rows in data.
	testRows= range(int(split*len(array_data)),int((split+.1)*len(array_data)))

	trainSet= np.delete(array_data, testRows,axis=0)#testrows in arraydata get deleted as testRow is the test set
	testSet=array_data[testRows]# getting the rest 10%

	k=20#  nieghbors per test set
	
	knn_final=[]
	
	for i in range(0, len(testSet)):
		new_kvalues=[]
		compare_euc= []
		#############going over training set
		for j in range(0, len(trainSet)):	
			euc_distance=  euclidean(testSet[i], trainSet[j])
			compare_euc.append((euc_distance,trainSet[j,-1],j))# this store index number, euclidean value and label
		#getting the least 3 euclidean
		compare_euc_sorted= sorted(compare_euc, key=lambda x: x[0])
		for l in range(k):#this stores only the first 3 values with minimum euclidean
			new_kvalues.append((compare_euc_sorted[l]))
		b=0
		c=0
		for m in new_kvalues:
			if m[1]== 0.0:
				b+=1
			elif m[1]==1.0:
				c+= 1
		if(b>c):
			knn_final.append(0.0)
		elif(c>b):
			knn_final.append(1.0)
		#print("going to equal")
		else:# write this in the report
			knn_final.append(new_kvalues[0][1])

	mTP,mTN,mFP,mFN = calculateStats(testSet[:,-1], knn_final)
	return getAccuracy(mTP,mTN,mFP,mFN),getPrecision(mTP,mTN,mFP,mFN), getRecall(mTP,mTN,mFP,mFN), getF_1Measure(mTP,mTN,mFP,mFN)


Acc_1=0
Prec_1=0
Rec_1 =0
F1_1= 0

#gettting average after cross validation
for i in range(0,10):
	Acc,Prec, Rec, F1=DataSet("project3_dataset1.txt", .1*i)
	print(Acc)
	Acc_1+= Acc
	Prec_1+= Prec
	Rec_1 += Rec
	F1_1 += F1
print("FInalllllllllll")
print((Acc_1/10)*100," ", (Prec_1/10*100)," ",(Rec_1/10*100), " ", (F1_1/10*100))
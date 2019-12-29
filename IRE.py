import math
import numpy as np

class IRE:
	def __init__(self):
		self.feature_pool = []
		self.labels = []


	def GetFeature(self,feature_map_list):
		feature_vector = []
		for feature_map in feature_map_list:
			feature_vector.append(self.MaxPooling(feature_map))
		return feature_vector


	def RecordFeature(self,feature_vector,label):
		self.feature_pool.append(feature_vector)
		self.labels.append(label)
		return


	def MaxPooling(self,feature_map):
		return np.max(feature_map.cpu().clone().detach().numpy())


	def ComputeSim(self,vector1,vector2):
		sum_mul = 0
		sum_square1 = 0
		sum_square2 = 0
		for i in range(len(vector1)):
			sum_mul += vector1[i]*vector2[i]
			sum_square1 += pow(vector1[i],2)
			sum_square2 += pow(vector2[i],2)
		return sum_mul/(math.sqrt(sum_square1)*math.sqrt(sum_square2))


	def RankImage(self,image_feature,k):
		target_vector = self.GetFeature(image_feature)

		sim = []
		for record_feature in self.feature_pool:
			sim.append(self.ComputeSim(target_feature,record_feature))

		rank = []
		for i in range(len(sim)):
			rank.append((sim[i],self.labels[i]))

		rank.sort(reverse = True)

		return rank[0:k]


	def RankFeature(self,target_feature,k):
		sim = []
		for record_feature in self.feature_pool:
			sim.append(self.ComputeSim(target_feature,record_feature))

		rank = []
		for i in range(len(sim)):
			rank.append((sim[i],self.labels[i]))

		rank.sort(reverse = True)

		return rank[0:k]


	def Poll(self,rank):
		result = [0]*12
		for member in rank:
			result[member[1]] += 1
		return result


	def Training(self,image_feature,image_label):
		feature_vector = self.GetFeature(image_feature)
		self.RecordFeature(feature_vector,image_label)
		return


	def Get_Info(self):
		features = np.array(self.feature_pool)
		labels = np.array(self.labels)
		return features,labels


	def Read_Info(self,path):
		print('Read information')
		self.feature_pool = np.load(path + '/features.npy').tolist()
		self.labels = np.load(path + '/labels.npy').tolist()
		print(len(self.feature_pool))
		print(len(self.labels))
		return
		


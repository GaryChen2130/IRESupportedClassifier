import math
import numpy as np

class IRE:
	def __init__(self):
		self.feature_pool = []
		self.feature_pool_validate = []
		self.feature_pool_test = []
		self.labels = []
		self.labels_validate = []
		self.labels_test = []


	def GetFeature(self,feature_map_list):
		feature_vector = []
		for feature_map in feature_map_list:
			feature_vector.append(self.MaxPooling(feature_map))
		return feature_vector


	def RecordFeature(self,feature_vector,label,state):
		if state == 0:
			self.feature_pool.append(feature_vector)
			self.labels.append(label)
		elif state == 1:
			self.feature_pool_validate.append(feature_vector)
			self.labels_validate.append(label)
		elif state == 2:
			self.feature_pool_test.append(feature_vector)
			self.labels_test.append(label)
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


	def Training(self,image_feature,image_label,state):
		feature_vector = self.GetFeature(image_feature)
		self.RecordFeature(feature_vector,image_label,state)
		return


	def Get_Info(self):
		features = np.array(self.feature_pool)
		features_validate = np.array(self.feature_pool_validate)
		features_test = np.array(self.feature_pool_test)
		labels = np.array(self.labels)
		labels_validate = np.array(self.labels_validate)
		labels_test = np.array(self.labels_test)
		return features,features_validate,features_test,labels,labels_validate,labels_test


	def Read_Info(self,path):
		print('Read information')
		self.feature_pool = np.load(path + '/features.npy').tolist()
		self.feature_pool_validate = np.load(path + '/features_validate.npy').tolist()
		self.feature_pool_test = np.load(path + '/features_test.npy').tolist()
		self.labels = np.load(path + '/labels.npy').tolist()
		self.labels_validate = np.load(path + '/labels_validate.npy').tolist()
		self.labels_test = np.load(path + '/labels_test.npy').tolist()

		print('Training Set')
		print(len(self.feature_pool))
		print(len(self.labels))
		print('Validating Set')
		print(len(self.feature_pool_validate))
		print(len(self.labels_validate))
		print('Testing Set')
		print(len(self.feature_pool_test))
		print(len(self.labels_test))

		return
		


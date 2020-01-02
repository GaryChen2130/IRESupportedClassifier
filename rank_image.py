import time
import numpy as np
from IRE import *

if __name__ == '__main__':
	IRE = IRE()
	IRE.Read_Info('.')

	classes = np.load('./classes.npy').tolist()
	classes_validate = np.load('./classes_validate.npy').tolist()
	classes_test = np.load('./classes_test.npy').tolist()
	print('classes')
	print(len(classes))
	print(len(classes_validate))
	print(len(classes_test))

	combine_data = []
	combine_data_validate = []
	combine_data_test = []

	start_time = time.time()

	k = 5

	for i in range(len(classes)):
		rank = IRE.RankFeature(IRE.feature_pool[i],k)
		IRE_result = IRE.Poll(rank)
		combine_data.append(classes[i] + IRE_result)

	for i in range(len(classes_validate)):
		rank = IRE.RankFeature(IRE.feature_pool_validate[i],k)
		IRE_result = IRE.Poll(rank)
		combine_data_validate.append(classes_validate[i] + IRE_result)

	acc = 0
	print('----------------Testing----------------')
	for i in range(len(classes_test)):
		rank = IRE.RankFeature(IRE.feature_pool_test[i],k)
		IRE_result = IRE.Poll(rank)
		combine_data_test.append(classes_test[i] + IRE_result)
		if IRE_result.index(max(IRE_result)) == IRE.labels_test[i]:
			acc += 1
		print('accuracy:' + str(acc/(i + 1)))
		print(IRE_result)
		print(IRE.labels_test[i])
		print()

	np.save('./combine_data',np.array(combine_data))
	np.save('./combine_data_validate',np.array(combine_data_validate))
	np.save('./combine_data_test',np.array(combine_data_test))

	end_time = time.time()
	print("--- %s sec ---" % (end_time - start_time))

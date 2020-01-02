import time
import numpy as np
from IRE import *

if __name__ == '__main__':
	IRE = IRE()
	IRE.Read_Info('.')

	classes_test = np.load('./classes_test.npy').tolist()
	print(len(classes_test))

	combine_data = []
	combine_data_validate = []
	combine_data_test = []

	k = 5
	acc = 0
	for i in range(len(classes_test)):
		rank = IRE.RankFeature(IRE.feature_pool_test[i],k)
		result = IRE.Poll(rank)
		if result.index(max(result)) == IRE.labels_test[i]:
			acc += 1
		print('accuracy:' + str(acc/(i + 1)))
		print(result)
		print(IRE.labels_test[i])
		print()
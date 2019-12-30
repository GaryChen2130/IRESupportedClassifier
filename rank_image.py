import numpy as np
from IRE import *

if __name__ == '__main__':
	IRE = IRE()
	IRE.Read_Info('.')
	classes = np.load('./classes.npy').tolist()
	combine_data = []
	print(len(classes))
	for i in range(len(classes)):
		rank = IRE.RankFeature(IRE.feature_pool[i],10)
		IRE_result = IRE.Poll(rank)
		combine_data.append(classes[i] + IRE_result)
	np.save('./combine_data',np.array(combine_data))
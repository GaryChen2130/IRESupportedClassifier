import numpy as np
import os
import sys
import time
import imutils
import math
import cv2
from skimage.measure import compare_ssim


def Compute_Sim(img0,img1):
	hist0 = cv2.calcHist([img0],[0],None,[256],[0.0,255.0])
	hist1 = cv2.calcHist([img1],[0],None,[256],[0.0,255.0])

	degree = 0
	for i in range(len(hist0)):
		if hist0[i] != hist1[i]:
			degree += (1 - abs(hist0[i] - hist1[i]) / max(hist0[i], hist1[i]))
		else:
			degree += 1

	return degree/len(hist0)


def Compare_Sim(img0,img1):
	
	# Method1 SSIM
	sub_img0 = cv2.split(img0)
	sub_img1 = cv2.split(img1)
	Sum = 0
	for ch0,ch1 in zip(sub_img0,sub_img1):
		(score,diff) = compare_ssim(ch0, ch1, full=True)
		Sum += score
	score = Sum/3

	'''
	# Method2: Histogram
	sub_img0 = cv2.split(img0)
	sub_img1 = cv2.split(img1)

	score = 0
	for ch0,ch1 in zip(sub_img0,sub_img1):
		score += Compute_Sim(ch0,ch1)
	score = score/3
	'''

	return score


def Poll(rank_list):
	result = [0]*12
	for element in rank_list:
		result[labels[element[1].split('_')[0]]] += 1
	return result

if __name__ == '__main__':
	labels = {
        'Abyssinian': 0,
        'Bengal': 1,
        'Birman': 2, 
        'Bombay': 3, 
        'British': 4,
        'Egyptian': 5,
        'Maine': 6, 
        'Persian': 7,
        'Ragdoll': 8,
        'Russian': 9,
        'Siamese': 10,
        'Sphynx': 11
    }

	INPUT_PATH = './dataset/crop_image/train/'
	files = os.listdir(INPUT_PATH)
	images = []
	for afile in files:
		if afile[-4:] == '.txt':
			continue
		images.append(cv2.imread(INPUT_PATH + afile))

	acc = 0
	k = 5
	for i in range(len(images)):
		rank_list = []
		for j in range(len(images)):
			if j == i:
				continue

			score = Compare_Sim(images[i],images[j])
			rank_list.append((score,files[j]))

		rank_list.sort(reverse=True)
		result = Poll(rank_list[0:k])

		if labels[files[i].split('_')[0]] == result.index(max(result)):
			acc += 1

		print('image ' + str(i) + ':')
		print(result)
		print('Most Similar: ' + str(result.index(max(result))))
		print('accuracy:' + str(acc/(i + 1)) + '\n')
			
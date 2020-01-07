#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 下午5:15
# @Author  : Aries
# @Site    :
# @File    : ItemSimilarity.py
# @Software: PyCharm
'''
计算推荐的相似度
'''
from collections import defaultdict
from operator import itemgetter

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

'''
生成数据
'''


def load_file(filename):
	f = open(filename, mode='r')
	for i, line in enumerate(f):
		yield line.strip('\r\n')
	f.close()


def generate_data(filename):
	source = {}
	for line in load_file(filename):
		userid, movieid, rating, _ = line.split('::')
		source.setdefault(userid, {})  # 初始化source
		source[userid][movieid] = int(rating)
	print('generate data:')
	print(source['1'])
	return source


'''
计算同现相似度
'''


def CooccurrenceSimilarity(source):
	#  辅助计算,计算movie N---被多少用户喜欢
	pref_each = {}
	coocu_fre_matrix = {}
	coocu_simi_matrix = {}
	for user, movies in source.items():
		for movie in movies:
			if movie not in pref_each:
				pref_each[movie] = 0
			pref_each[movie] += 1
	# 接下来实现频次共现矩阵
	for user, movies in source.items():
		for m1 in movies:
			coocu_fre_matrix.setdefault(m1, defaultdict(int))
			for m2 in movies:
				if m1 == m2:
					continue
				coocu_fre_matrix[m1][m2] += 1
	# 接着实现相似度共现矩阵:
	for m1, relative in coocu_fre_matrix.items():
		coocu_simi_matrix.setdefault(m1, {})
		for m2, count in relative.items():
			coocu_simi_matrix[m1][m2] = count / np.sqrt(pref_each[m1] * pref_each[m2])
	
	return coocu_simi_matrix


'''
计算欧几里得相似度
'''


def EnclideanSimilarity(source):
	coocu_fre_matrix = {}
	# 接下来实现频次共现矩阵
	for user, movies in source.items():
		for m1 in movies:
			coocu_fre_matrix.setdefault(m1, defaultdict(int))
			for m2 in movies:
				if m1 == m2:
					continue
				coocu_fre_matrix[m1][m2] += 1
	enclidean_dis_matrix = {}
	for user, movies in source.items():
		for mI, ratingI in movies.items():
			enclidean_dis_matrix.setdefault(mI, defaultdict(int))
			for mJ, ratingJ in movies.items():
				if mI == mJ:
					continue
				enclidean_dis_matrix[mI][mJ] += (ratingI - ratingJ) * (ratingI - ratingJ)
	enclidean_simi_matrix = {}
	for m1, relative in coocu_fre_matrix.items():
		enclidean_simi_matrix.setdefault(m1, {})
		for m2, count in relative.items():
			enclidean_simi_matrix[m1][m2] = count / (1 + np.sqrt(enclidean_dis_matrix[m1][m2]))
	return enclidean_simi_matrix


'''
计算余弦相似度  <Cosine相似度>
'''


def CosineSimilarity(source):
	# cosine下计算  x*y x*x y*y
	cosine_dis_matrix = {}
	for user, movie in source.items():
		for mI, ratingI in movie.items():
			cosine_dis_matrix.setdefault(mI, {})
			for mJ, ratingJ in movie.items():
				cosine_dis_matrix[mI].setdefault(mJ, [0, 0, 0])
				tmp_list = cosine_dis_matrix[mI][mJ]
				tmp_list[0] += ratingI * ratingJ
				tmp_list[1] += ratingI * ratingI
				tmp_list[2] += ratingJ * ratingJ
				cosine_dis_matrix[mI][mJ] = tmp_list
	cosine_simi_matrix = {}
	for mI, relative in cosine_dis_matrix.items():
		cosine_simi_matrix.setdefault(mI, {})
		for mJ, tmp in relative.items():
			cosine_simi_matrix[mI][mJ] = tmp[0] / (np.sqrt(tmp[1]) * np.sqrt(tmp[2]))
	return cosine_simi_matrix


'''
进行物品的推荐
'''


def recommend(source, coocu_simi_matrix, userid):
	K = 20
	N = 10
	rank = {}
	# 根据userid获取当前用户对物品的评分
	movies = source[userid]
	for movie, rating in movies.items():
		# 根据当前movie获取关联movie
		for related_movie, similar in sorted(coocu_simi_matrix[movie].items(), key=itemgetter(1), reverse=True)[:K]:
			if related_movie in movie:
				continue
			rank.setdefault(related_movie, 0)
			rank[related_movie] += rating * similar
	return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 下午3:28
# @Author  : Aries
# @Site    : 
# @File    : main_itemcf.py
# @Software: PyCharm
import ItemSimilarity as similarTool
import tensorflow as  tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../ml-1m/ratings.dat', """where is data""")
tf.app.flags.DEFINE_string('simi_type', 'coocu', """support coocu(同现相似度) enclidean(欧几里得相似度) cosine(余弦相似度)""")


def main():
	'''
	启动入口
	:return:
	'''
	source = similarTool.generate_data(FLAGS.train_dir)
	if FLAGS.simi_type == 'coocu':
		simi_matrix = similarTool.CooccurrenceSimilarity(source)
	elif FLAGS.simi_type == 'enclidean':
		simi_matrix = similarTool.EnclideanSimilarity(source)
	elif FLAGS.simi_type == 'cosine':
		simi_matrix = similarTool.CosineSimilarity(source)
	result = similarTool.recommend(source, simi_matrix, '10')
	print(result)


if __name__ == '__main__':
	main()

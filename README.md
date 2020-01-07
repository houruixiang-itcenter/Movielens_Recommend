协同过滤算法:
---
#### 支持python版本
使用py 3.x 使用2.x微调print等少部分函数即可

#### 数据源下载
数据源: http://files.grouplens.org/datasets/movielens/ml-1m.zip
解压到任意位置指明即可

#### 运行代码
以itemCF为例(可以基于此类比userCF)
python main_itemcf.py --train_dir ml-1m/ratings.dat --simi_type enclidean
或者pycharm右键run Configurations添加上述两个params


--- train_dir:数据源路径

---simi_type:使用何种相似度进行计算  目前支持 coocu(同现相似度) enclidean(欧几里得相似度) cosine(余弦相似度)


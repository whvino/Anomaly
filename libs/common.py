import os
import tensorflow as tf

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def checkrank(tensor, ranks):
    for rank in ranks:   #rank 张量的秩  张量的秩是唯一选择张量的每个元素所需的索引的数量，比如三维张量的秩就是3
        if tf.rank(tensor) != rank:
            raise Exception('the rank of tensor {} is not equal to anyone of {}'.format(tensor, rank))
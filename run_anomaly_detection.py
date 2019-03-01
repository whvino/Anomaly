import os
import sys
import numpy as np
import yaml
import tensorflow as tf
from libs.sista_rnn_anomaly_detection import sista_rnn_anomaly_detection_TSC, sista_rnn_anomaly_detection_AE
from libs.common import checkdir
from libs import FLAGS
import multiprocessing

def main(config):
    if sys.argv[2] == '0':    #training
        #config是读取sys.argv[1]后的字典，存储 xxx.yaml里面的配置信息    yaml也是键值对形式
        # >> > "{} {}".format("hello", "world")  # 不设置指定位置，按默认顺序
        # 'hello world'
        #
        # >> > "{0} {1}".format("hello", "world")  # 设置指定位置
        # 'hello world'
        #
        # >> > "{1} {0} {1}".format("hello", "world")  # 设置指定位置
        # 'world hello world'
        config['train_videos_txt'] = config['train_videos_txt'].format(config['dataset'])
        config['train_feature_path'] = config['train_feature_path'].format(config['dataset'], config['twostream_model'])
        config['prefix'] = config['prefix'].format(config['dataset'], config['twostream_model'], config['model_type'],
                                                   config['K'], config['n_hidden'], config['lambda1'], config['lambda2'], config['time_steps'])
        config['log_path'] = os.path.join(config['log_path'], config['prefix'])
        checkdir(config['ckpt_path'])    #检查目录是否存在，不存在则新建
        checkdir(config['log_path'])
    elif sys.argv[2] == '1':    #testing
        config['time_steps'] = 1
        config['batch_size'] = 1
        config['test_videos_txt'] = config['test_videos_txt'].format(config['dataset'])
        config['test_feature_path'] = config['test_feature_path'].format(config['dataset'], config['twostream_model'])
        config['prefix'] = config['prefix'].format(config['dataset'], config['twostream_model'], config['model_type'],
                                                   config['K'], config['n_hidden'], config['lambda1'], config['lambda2'], 10)
        checkdir(config['save_results_path'])
        config['save_results_path'] = os.path.join(config['save_results_path'], config['prefix'])
    else:
        raise Exception('only support 0 for training or 1 for testing')
    rng = np.random.RandomState(config['seed'])   #随机数生成，用种子 config['seed'],只要种子一样，每次生成的随机数序列就一样

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    tf_config = tf.ConfigProto()   #tf.ConfigProto()用来配置session运行参数
    tf_config.gpu_options.allow_growth = True    #动态申请显存，用多少申请多少
    sess = tf.Session(config=tf_config)

    # ======================== SISTA-RNN ============================= #
    print('... building the sista-rnn networks')
    #placeholder是占位，在计算图中占据相应内存，sess运行时才feed_dict喂入数据
    pre_input = tf.placeholder(tf.float32, [1, config['batch_size'] * 21, config['n_input']])    #dtype,shape
    now_input = tf.placeholder(tf.float32, [config['time_steps'], config['batch_size'] * 21, config['n_input']])
    A = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / (config['n_input'] + config['n_hidden'])),    #np.sqrt是平方根
            high=np.sqrt(6.0 / (config['n_input'] + config['n_hidden'])),
            size=(config['n_input'], config['n_hidden'])
        ) / 2.0,   #数组元素值变成一半
        dtype=np.float32
    )

    if config['model_type'] == FLAGS.TSC:
        model = sista_rnn_anomaly_detection_TSC([pre_input, now_input], None, A, sess, config)
    elif config['model_type'] == FLAGS.AE:
        model = sista_rnn_anomaly_detection_AE([pre_input, now_input], None, A, sess, config)
    else:
        raise Exception('not support {}, only support TSC and AE model'.format(config['model_type']))

    if sys.argv[2] == '0':
        model.train()
    else:
        model.test()

if __name__ == '__main__':
    # ==========================load config================================ #

    if len(sys.argv) < 3:    #sys.argv是一个列表 argv[0]是程序名比如xxx.py   我们运行程序的时候传入外部参数如  xxx.py a b c d,  那么argv[1] = a, argv[2] = b...
        raise Exception('usage: python xxx.py config/xxx.yaml 0/1 (0 is training, 1 is testing)')     #在运行此程序时，应传入外部参数 xxx.py config/xxx.yaml 0/1
    with open(sys.argv[1], 'r') as stream:
        config = yaml.load(stream)

    main(config)
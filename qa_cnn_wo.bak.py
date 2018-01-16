#!/usr/bin/python
#  -*- coding: utf-8 -*-

#basic cnn for qa

import random
import operator
import time
import os, sys,cPickle

# sys.path.append('/home/zhengxing/QAModel')

import codecs
import numpy as np
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logUtil import log
from timeUtil import GetNowTime


#reload(sys)
#sys.setdefaultencoding('utf-8')

theano.config.floatX = 'float32'

#TODO change path to your dataset
'''
#local
trainfile = 'D:\\xxx0624\\insuranceQA\\corpus\\nlpcc16-entity-data\\train_1'
train0file = 'D:\\xxx0624\\insuranceQA\\corpus\\nlpcc16-entity-data\\train_0'
test1file = 'D:\\xxx0624\\insuranceQA\\corpus\\nlpcc16-entity-data\\test'
vectorsfile = 'D:\\xxx0624\\Data\\word_vector_cn\\baike_vector.bin'
'''

'''
#server
file_tag = 'movie'
trainfile = '../../corpus/nlpcc16-entity-data/'+file_tag+'_train_1_detail'
train0file = '../../corpus/nlpcc16-entity-data/'+file_tag+'_train_0_detail'
test1file = '../../corpus/nlpcc16-entity-data/'+file_tag+'_test'
vectorsfile = '/opt/exp_data/word_vector_cn/baike_vector.bin'
'''

version_num = '1'
trainfile = '/home/liuxiao/tongji/train1_data_version_' + version_num
train0file = '/home/liuxiao/tongji/train0_data_version_' + version_num
test1file = '/home/liuxiao/tongji/test_data_version_' + version_num
vectorsfile = '/home/liuxiao/tongji/vectors.bin'
paramsfile = '/home/liuxiao/tongji/cnn/parameters'
validationfile = '/home/liuxiao/tongji/nlpcc_validation'
#vectorsfile = '/opt/exp_data/word_vector_cn/all_dbqa.bin'

#log file
logfile_path = '/home/liuxiao/tongji/cnn'


###########################################################
# read qa data
###########################################################
def build_vocab():
    global trainfile
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(trainfile):
        items = line.strip().split('\t')
        for i in range(2, 4):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    global train0file
    for line in open(train0file):
        items = line.strip().split('\t')
        for i in range(2, 4):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab


def build_idf_vocab(idf_file_path):
    idf_vocab = {}
    with codecs.open(idf_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) == 2:
                w = line[0]
                idf = line[1]
                if w not in idf_vocab:
                    idf_vocab[w] = float(idf)
    return idf_vocab


def load_vectors():
    global vectorsfile
    #w2v = Word2Vec.load_word2vec_format(vectorsfile, binary=True)
    w2v = KeyedVectors.load_word2vec_format(vectorsfile, binary=True)
    return w2v


def get_vector_of_dim(w2v, word, dim):
    if word.decode('utf-8')  in w2v.vocab:
        v_list = w2v[word.decode('utf-8')].tolist()
        if dim > len(v_list):
            for i in range(len(v_list), dim, 1):
                v_list.append(0.01)
        else:
            v_list = v_list[:dim]
        return v_list
    else:
        v_list = []
        for i in range(0, dim):
            v_list.append(0.01)
        return v_list


def load_word_embeddings(vocab, embedding_size):
    w2v = load_vectors()
    embeddings = [] #brute initialization
    #print 'vocab size = ', len(vocab)
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, embedding_size):
            vec.append(0.01)
        embeddings.append(vec)
    print 'vocab size = ', len(embeddings)
    for word, code in vocab.items():
        embeddings[code] = get_vector_of_dim(w2v, word, embedding_size)
    return np.array(embeddings, dtype=theano.config.floatX)


#w2v_dim is w2v's dim
#idf_dim is idf's dim(now, idf_dim is 1)
def get_vector_of_dim_with_idf(w2v, word, w2v_dim, idf_dim, idf_vocab):
    if word.decode('utf-8')  in w2v.vocab:
        v_list = w2v[word.decode('utf-8')].tolist()
        if w2v_dim > len(v_list):
            for i in range(len(v_list), w2v_dim, 1):
                v_list.append(0.01)
        else:
            v_list = v_list[:w2v_dim]
    else:
        v_list = []
        for i in range(0, w2v_dim):
            v_list.append(0.01)
    if word in idf_vocab:
        v_list.append(idf_vocab[word])
    else:
        v_list.append(0.01)
    return v_list


def load_word_embeddings_with_idf(vocab, embedding_size, idf_vocab):
    w2v = load_vectors()
    embeddings = [] #brute initialization
    #print 'vocab size = ', len(vocab)
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, embedding_size):
            vec.append(0.01)
        embeddings.append(vec)
    print 'vocab size = ', len(embeddings)
    for word, code in vocab.items():
        embeddings[code] = get_vector_of_dim_with_idf(w2v, word, embedding_size - 1, 1, idf_vocab)
    return np.array(embeddings, dtype=theano.config.floatX)


#be attention initialization of UNKNNOW
def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x


def load_train_list():
    global trainfile
    trainList = []
    for line in open(trainfile):
        trainList.append(line.strip().split('\t'))
    return trainList


def load_test_list():
    global test1file
    testList = []
    for line in open(test1file):
        testList.append(line.strip().split('\t'))
    return testList

def load_validation_list():
    global validationfile
    validationList = []
    for line in open(validationfile):
        validationList.append(line.strip().split('\t'))
    return validationList


#load data from train file
#the wrong answer is random
def load_train_data(trainList, vocab, batch_size, words_num_dim):
    train_1, train_2, train_3 = [], [], []
    for i in range(0, batch_size):
        pos = trainList[random.randint(0, len(trainList)-1)]
        neg = trainList[random.randint(0, len(trainList)-1)]
        train_1.append(encode_sent(vocab, pos[2], words_num_dim))
        train_2.append(encode_sent(vocab, pos[3], words_num_dim))
        train_3.append(encode_sent(vocab, neg[3], words_num_dim))
    return np.array(train_1, dtype=theano.config.floatX), np.array(train_2, dtype=theano.config.floatX), np.array(train_3, dtype=theano.config.floatX)


class myDict(object):
    def __init__(self, v_list):
        self.v = v_list


def load_train0_dict():
    global train0file
    train0Dict = {}
    with open(train0file) as f:
        for line in f:
            line = line.strip().split('\t')
            qid = int(line[1].split(":")[-1])
            if qid not in train0Dict:
                train0Dict[qid] = myDict([])
            train0Dict[qid].v.append(line[3])
    return train0Dict


def check_int(id):
    id = str(id)
    for i in id:
        if i < '0' or i > '9':
            return False
    return True

# def gener_overlap(queList, ansList):


#load data from 2 train files
#the wrong anser is from the file named '*0'
def load_train_data_from_2files(train0Dict, train1List, vocab, batch_size, words_num_dim):
    train_1, train_2, train_3 = [], [], []
    cnt = 0
    while True:
        index = random.randint(0, len(train1List)-1)
        pos = train1List[index]
        qid = pos[1].strip().split(":")[-1]
        #exist some bug data
        if check_int(qid) == False:
            continue
        qid = int(qid)
        neg = None
        if qid in train0Dict:
            neg = train0Dict[qid].v[(random.randint(0, len(train0Dict[qid].v) - 1))]
        else:
            #if this question dont have neg answer
            neg = train1List[random.randint(0, len(train1List)-1)][3]
        train_1.append(encode_sent(vocab, pos[2], words_num_dim))
        train_2.append(encode_sent(vocab, pos[3], words_num_dim))
        train_3.append(encode_sent(vocab, neg, words_num_dim))
        cnt += 1
        if cnt >= batch_size:
            break
    return np.array(train_1, dtype=theano.config.floatX), np.array(train_2, dtype=theano.config.floatX), np.array(train_3, dtype=theano.config.floatX)


def load_test_data(testList, vocab, index, batch_size, words_num_dim):
    x1, x2, x3 = [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        items = testList[true_index]
        x1.append(encode_sent(vocab, items[2], words_num_dim))
        x2.append(encode_sent(vocab, items[3], words_num_dim))
        x3.append(encode_sent(vocab, items[3], words_num_dim))
    return np.array(x1, dtype=theano.config.floatX), np.array(x2, dtype=theano.config.floatX), np.array(x3, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def gener_wordoverlap(ques_matrix, ans_matrix, batch_size, sentence_len, vocab):
    ques_wordoverlap = []
    ans_wordoverlap = []
    for i in range(batch_size):
        ques_each_wo = []
        ans_each_wo = []
        for j in range(sentence_len):
            similarity_ques = 0.0
            similarity_ans = 0.0
            for k in range(sentence_len):
                if(ques_matrix[i][j] == ans_matrix[i][k] and ques_matrix[i][j] != vocab['<a>']):
                    similarity_ques = 1.0
                    break
            for l in range(sentence_len):
                if(ans_matrix[i][j] == ques_matrix[i][l] and ans_matrix[i][j] != vocab['<a>']):
                    similarity_ans = 1.0
                    break
            ques_each_wo.append(similarity_ques)
            ans_each_wo.append(similarity_ans)
        ques_wordoverlap.append(ques_each_wo)
        ans_wordoverlap.append(ans_each_wo)
    return np.asarray(ques_wordoverlap, dtype=theano.config.floatX),np.asarray(ans_wordoverlap, dtype=theano.config.floatX)

def save_model_parameters_theano(model, outfile, filter_sizes):
    save_file = open(outfile, 'wb')
    tparams = model.tparams

    for filter_size in filter_sizes:
        cPickle.dump(tparams[_p('cnn_W', str(filter_size))].get_value(borrow=True), save_file, -1)
        cPickle.dump(tparams[_p('cnn_b', str(filter_size))].get_value(borrow=True), save_file, -1)

    cPickle.dump(tparams[_p('lookup','table')].get_value(borrow=True), save_file, -1)
    # cPickle.dump(tparams[_p('softmax', 'W')].get_value(borrow=True), save_file, -1)
    # cPickle.dump(tparams[_p('softmax', 'b')].get_value(borrow=True), save_file, -1)

    # cPickle.dump(tparams[_p('leaner_transform', 'W')].get_value(borrow=True), save_file, -1)

    print "Saved model parameters to %s." % outfile
    save_file.close()

def load_model_parameters(model, path, filter_sizes):
    tparams = model.tparams
    save_file = open(path, 'rb')

    for filter_size in filter_sizes:
        tparams[_p('cnn_W', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)
        tparams[_p('cnn_b', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)

    tparams[_p('lookup','table')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('softmax', 'W')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('softmax', 'b')].set_value(cPickle.load(save_file), borrow=True)

    # tparams[_p('leaner_transform', 'W')].set_value(cPickle.load(save_file), borrow=True)

    print "Building model model from %s " % (path)
    save_file.close()

def validation(validate_model, testList, vocab, batch_size, words_num_dim):
    index, score_list = int(0), []
    while True:
        x1, x2, x3 = load_test_data(testList, vocab, index, batch_size, words_num_dim)
        batch_scores, nouse = validate_model(x1, x2, x3, 1.0)
        for score in batch_scores:
            if len(score_list) < len(testList):
                score_list.append(score)
            else:
                break
        index += batch_size
        #log('Evalution' + str(index), logfile_path)
        print 'Evalution...', str(index)
        if index >= len(testList):
            break
    sdict, index = {}, int(0)
    qa_count = 0
    for items in testList:
        qid = items[1].split(':')[1]
        question = items[2].strip('_<a>')
        answer = items[3].strip('_<a>')
        if not qid in sdict:
            sdict[qid] = []
        sdict[qid].append((score_list[index], items[0], question, answer))
        index += 1
        if int(qid) > qa_count:
            qa_count = int(qid)
    qa_count += 1
    top1 = float(0)
    map_sum = float(0)
    mrr_sum = float(0)
    #qid_count = 0

    for qid, items in sdict.items():
        items.sort(key=operator.itemgetter(0), reverse=True)
        #just for analysis
        global logfile_path
        analysis_log_file_path = logfile_path + '.analysis'
        for score, flag, question, answer in items:
            log('[' + str(qid) + ']' + question, analysis_log_file_path)
            log('[Predicted][' + '1:' + str(len(items)) + '] '
                + answer
                , analysis_log_file_path)
            break
        expected_answer_index = 0
        expected_answer_flag = False
        for score, flag, question, answer in items:
            expected_answer_index += 1
            if flag == '1':
                log('[Expected][' + str(expected_answer_index) + ':' + str(len(items)) + '] '
                    + answer
                    , analysis_log_file_path)
                expected_answer_flag = True
                break
        if expected_answer_flag == False:
            log('[Expected][' + str(qid) + '/' + flag + '/' + 'Not Exist!'
                    , analysis_log_file_path)
        log('', analysis_log_file_path)
        #for top1
        score, flag, question, answer = items[0]
        if flag == '1':
            top1 += 1
        #for mrr
        mrr_index = 0
        for score, flag, question, answer in items:
            mrr_index += 1
            if flag == '1':
                mrr_sum += float(1) / float(mrr_index)
        #for map
        map_index_down = 0
        map_index_up = 0
        temp_map_sum = 0
        for score, flag, question, answer in items:
            #log('[debug]' + ' qid=' + str(qid) + ' score=' + str(score) + ' flag=' + str(flag))
            map_index_down += 1
            if flag == '1':
                map_index_up += 1
                temp_map_sum += float(map_index_up) / float(map_index_down)
        temp_map_sum /= float(map_index_up)
        map_sum += temp_map_sum
        #log('qid = ' + str(qid) + ' / top1 count = ' + str(top1), logfile_path)
    top1 /= float(qa_count)
    mrr_sum /= float(qa_count)
    map_sum /= float(qa_count)
    log('top-1 = ' + str(top1) + ' / ' + 'mrr = ' + str(mrr_sum) + ' / ' + 'map = ' + str(map_sum), logfile_path)
    return mrr_sum


class QACnn(object):
    def __init__(self,word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters, margin_size, learning_rate):
        rng = np.random.RandomState(23455)
        self.params, tparams = [], {}

        input1, input2, input3 = T.matrix('x1'), T.matrix('x2'), T.matrix('x3')
        keep_prob = T.fscalar('keep_prob')

        lookup_table = theano.shared(word_embeddings)
        self.params += [lookup_table]
        tparams['lookup_table'] = lookup_table
        self.tparams = tparams
        #input1-问题, input2-正向答案, input3-负向答案
        #将每个字替换成字向量
        input_matrix1 = lookup_table[T.cast(input1.flatten(), dtype="int32")]
        input_matrix2 = lookup_table[T.cast(input2.flatten(), dtype="int32")]
        input_matrix3 = lookup_table[T.cast(input3.flatten(), dtype="int32")]

        #CNN的输入是4维矩阵，这里只是增加了一个维度而已
        input_x1 = input_matrix1.reshape((batch_size, 1, sequence_len, embedding_size))
        input_x2 = input_matrix2.reshape((batch_size, 1, sequence_len, embedding_size))
        input_x3 = input_matrix3.reshape((batch_size, 1, sequence_len, embedding_size))
        #print(input_x1.shape.eval())
        self.dbg_x1 = input_x1

        #sigmod function
        ix = T.matrix('ix')
        ip = 1 / (1 + T.exp(-ix))
        sig = theano.function([ix], ip)

        outputs_1, outputs_2, outputs_3 = [], [], []
        #设置多种大小的filter
        for filter_size in filter_sizes:
            #每种大小的filter的数量是num_filters
            filter_shape = (num_filters, 1, filter_size, embedding_size)
            image_shape = (batch_size, 1, sequence_len, embedding_size)
            fan_in = np.prod(filter_shape[1:])
            fan_out = filter_shape[0] * np.prod(filter_shape[2:])
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

            #卷积+max_pooling
            conv_out = conv2d(input=input_x1, filters=W, filter_shape=filter_shape, input_shape=image_shape)
            #卷积后的向量的长度为dss
            pooled_out = pool.pool_2d(input=conv_out, ws=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
            outputs_1.append(pooled_active)

            conv_out = conv2d(input=input_x2, filters=W, filter_shape=filter_shape, input_shape=image_shape)
            pooled_out = pool.pool_2d(input=conv_out, ws=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
            outputs_2.append(pooled_active)

            conv_out = conv2d(input=input_x3, filters=W, filter_shape=filter_shape, input_shape=image_shape)
            pooled_out = pool.pool_2d(input=conv_out, ws=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
            outputs_3.append(pooled_active)

            self.params += [W, b]
            self.dbg_conv_out = conv_out.shape
            self.tparams['cnn_W_' + str(filter_size)] = W
            self.tparams['cnn_b_' + str(filter_size)] = b

        num_filters_total = num_filters * len(filter_sizes)
        self.dbg_outputs_1 = outputs_1[0].shape
        #每一个句子的语义表示向量的长度为num_filters_total
        output_flat1 = T.reshape(T.concatenate(outputs_1, axis=1), [batch_size, num_filters_total])
        output_flat2 = T.reshape(T.concatenate(outputs_2, axis=1), [batch_size, num_filters_total])
        output_flat3 = T.reshape(T.concatenate(outputs_3, axis=1), [batch_size, num_filters_total])
        #dropout, keep_prob为1表示不进行dropout
        output_drop1 = self._dropout(rng, output_flat1, keep_prob)
        output_drop2 = self._dropout(rng, output_flat2, keep_prob)
        output_drop3 = self._dropout(rng, output_flat3, keep_prob)

        #计算问题和答案之前的向量夹角
        #计算向量的长度
        len1 = T.sqrt(T.sum(output_drop1 * output_drop1, axis=1))
        len2 = T.sqrt(T.sum(output_drop2 * output_drop2, axis=1))
        len3 = T.sqrt(T.sum(output_drop3 * output_drop3, axis=1))
        #计算向量之间的夹角
        cos12 = T.sum(output_drop1 * output_drop2, axis=1) / (len1 * len2)
        self.cos12 = cos12
        cos13 = T.sum(output_drop1 * output_drop3, axis=1) / (len1 * len3)
        self.cos13 = cos13

        zero = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)
        margin = theano.shared(np.full(batch_size, margin_size, dtype=theano.config.floatX), borrow=True)
        #Loss损失函数
        diff = T.cast(T.maximum(zero, margin - cos12 + cos13), dtype=theano.config.floatX)
        self.cost = T.sum(diff, acc_dtype=theano.config.floatX)
        #mini-batch数据的准确率(如果正向答案和问题之间的cosine大于负向答案和问题的cosine，则认为正确，
        #否则是错误的)
        #Loss和Accuracy是用来评估训练中模型时候收敛的两个很重要的指标
        self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)

        params = self.params
        grads = T.grad(self.cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [input1, input2, input3, keep_prob],
            [self.cost, self.accuracy, self.dbg_x1, self.dbg_outputs_1],
            updates=updates,

        )
        self.validate_model = theano.function(
            inputs=[input1, input2, input3, keep_prob],
            outputs=[cos12, cos13],
        )

    def _dropout(self, rng, layer, keep_prob):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(123456))
        mask = srng.binomial(n=1, p=keep_prob, size=layer.shape)
        output = layer * T.cast(mask, theano.config.floatX)
        output = output / keep_prob
        return output


def train():
    global logfile_path
    global trainfile
    global train0file
    global test1file

    batch_size = int(256)
    filter_sizes = [1,2,3]
    num_filters = 500
    words_num_dim = 50
    embedding_size = 300
    learning_rate = 0.001
    n_epochs = 20000
    validation_freq = 100
    keep_prob_value = 0.75
    margin_size = 0.05

    logfile_name = 'CNN-' + GetNowTime() + '-' \
                   + 'batch_size-' + str(batch_size) + '-' \
                   + 'num_filters-' + str(num_filters) + '-' \
                   + 'embedding_size-' + str(embedding_size) + '-' \
                   + 'n_epochs-' + str(n_epochs) + '-' \
                   + 'freq-' + str(validation_freq) + '-' \
                   + '-log.txt'
    logfile_path = os.path.join(logfile_path, logfile_name)
    os.mknod(logfile_name)

    log("New start ...", logfile_path)
    log(str(time.asctime(time.localtime(time.time()))), logfile_path)
    log("batch_size = " + str(batch_size), logfile_path)
    log("filter_sizes = " + str(filter_sizes), logfile_path)
    log("num_filters = " + str(num_filters), logfile_path)
    log("embedding_size = " + str(embedding_size), logfile_path)
    log("learning_rate = " + str(learning_rate), logfile_path)
    log("n_epochs = " + str(n_epochs), logfile_path)
    log("margin_size = " + str(margin_size), logfile_path)
    log("words_num_dim = " + str(words_num_dim), logfile_path)
    log("validation_freq = " + str(validation_freq), logfile_path)
    log("keep_prob_value = " + str(keep_prob_value), logfile_path)
    log("train_1_file = " + str(trainfile.split('/')[-1]), logfile_path)
    log("train_0_file = " + str(train0file.split('/')[-1]), logfile_path)
    log("test_file = " + str(test1file.split('/')[-1]), logfile_path)
    log("vector_file = " + str(vectorsfile.split('/')[-1]), logfile_path)

    vocab = build_vocab()
    #word_embeddings is list, shape = numOfWords*100
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    testList = load_test_list()
    train0Dict = load_train0_dict()

    model = QACnn(
        word_embeddings=word_embeddings,
        batch_size=batch_size,
        sequence_len=words_num_dim,
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        margin_size=margin_size,
        learning_rate=learning_rate)

    epoch = 0
    done_looping = False
    best_result = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #train_x1, train_x2, train_x3 = load_train_data(trainList, vocab, batch_size)
        train_x1, train_x2, train_x3 = load_train_data_from_2files(train0Dict, trainList, vocab, batch_size, words_num_dim)
        #print train_x3.shape
        cost_ij, acc, dbg_x1, dbg_outputs_1 = model.train_model(train_x1, train_x2, train_x3, keep_prob_value)
        log('load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', acc:' + str(acc), logfile_path)
        if epoch % validation_freq == 0:
            log('Evaluation ......', logfile_path)
            MRR_result = validation(model.validate_model, testList, vocab, batch_size, words_num_dim)
            if (MRR_result > best_result):
                save_model_parameters_theano(model, paramsfile, filter_sizes)
                best_MRR = MRR_result
                print 'best_MRR .............', best_MRR
        #print dbg_outputs_1

def predict(modelClass=QACnn):
    global logfile_path
    global trainfile
    global train0file
    global test1file
    global validationfile

    batch_size = int(256)
    filter_sizes = [1,2,3]
    num_filters = 500
    words_num_dim = 50
    embedding_size = 300
    learning_rate = 0.001
    n_epochs = 20000
    validation_freq = 100
    keep_prob_value = 0.75
    margin_size = 0.05

    logfile_name = 'CNN-' + GetNowTime() + '-' \
                   + 'batch_size-' + str(batch_size) + '-' \
                   + 'num_filters-' + str(num_filters) + '-' \
                   + 'embedding_size-' + str(embedding_size) + '-' \
                   + 'n_epochs-' + str(n_epochs) + '-' \
                   + 'freq-' + str(validation_freq) + '-' \
                   + '-log.txt'
    logfile_path = os.path.join(logfile_path, logfile_name)
    os.mknod(logfile_name)

    vocab = build_vocab()
    # word_embeddings is list, shape = numOfWords*100
    word_embeddings = load_word_embeddings(vocab, embedding_size)

    model = modelClass(
        word_embeddings=word_embeddings,
        batch_size=batch_size,
        sequence_len=words_num_dim,
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        margin_size=margin_size,
        learning_rate =learning_rate)

    load_model_parameters(model, paramsfile, filter_sizes)

    print 'start'
    validationList = load_validation_list()
    print 'Evaluation ......'
    validation(model.validate_model, validationList, vocab, batch_size, words_num_dim)
    print 'end'


if __name__ == '__main__':
    train()
    # predict()


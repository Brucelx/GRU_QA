#-*-coding:utf-8-*-
#!/usr/bin/env python

import numpy as np
import sys
import os, sys, timeit, random, operator

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import memory_profiler
import pickle

reload(sys)
sys.setdefaultencoding( "utf-8" )
theano.config.floatX = 'float32'

trainfile = unicode('E:/git/GRU_QA/GRU_QA/nlpcc_train','utf-8')
testfile = unicode('E:/git/GRU_QA/GRU_QA/nlpcc_test','utf-8')
vectorsfile = unicode('E:/git/GRU_QA/GRU_QA/vectors.bin','utf-8')
resultfile = unicode('E:/git/GRU_QA/GRU_QA/output.txt','utf-8')
trainfile_linux = unicode('/home/liuxiao/GRU/nlpcc_train','utf-8')
testfile_linux = unicode('/home/liuxiao/GRU/nlpcc_test','utf-8')
vectorsfile_linux = unicode('/home/liuxiao/GRU/vectors.bin','utf-8')
resultfile_linux = unicode('/home/liuxiao/GRU/output.txt','utf-8')

if(os.path.exists('/home/liuxiao/GRU/nlpcc_train')):
    trainfile = trainfile_linux
    testfile = testfile_linux
    vectorsfile = vectorsfile_linux
    resultfile = resultfile_linux

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
    return vocab

def load_vectors(embedding_size):
    global vectorsfile
    vectors = {}
    for line in open(vectorsfile):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = []
        for i in range(1, embedding_size+1):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, embedding_size):
    vectors = load_vectors(embedding_size)
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, embedding_size):
            vec.append(0.0)
        embeddings.append(vec)
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
    return np.array(embeddings, dtype='float32')


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
    question_answer = []
    qid = 'qid:0'
    for line in open(trainfile):
        one_qa = line.strip().split('\t')
        if (one_qa[1] == qid):
            question_answer.append(one_qa)
        else:
            trainList.append(question_answer)
            qid = one_qa[1]
            question_answer = []
            question_answer.append(one_qa)
    trainList.append(question_answer)
    return trainList

def load_train_list2():
    global trainfile
    trainList = []
    for line in open(trainfile):
        trainList.append(line.strip().split('\t'))
    return trainList

def load_test_list():
    global testfile
    testList = []
    question_answer = []
    qid = 'qid:0'
    for line in open(testfile):
        one_qa = line.strip().split('\t')
        if (one_qa[1] == qid):
            question_answer.append(one_qa)
        else:
            testList.append(question_answer)
            qid = one_qa[1]
            question_answer = []
            question_answer.append(one_qa)
    testList.append(question_answer)
    return testList

def load_test_list2():
    global testfile
    testList = []
    for line in open(testfile):
        testList.append(line.strip().split('\t'))
    return testList

def load_data(trainList, vocab, batch_size):
    train_q, train_ra, train_ea = [], [], []
    padding = []
    for i in range(0,100):
        padding.append(15)
    for i in range(0, batch_size):
        question_answer = trainList[random.randint(0, len(trainList)-1)]
        error_answer = []
        question_only = True
        for qa_pair in question_answer:
            if(qa_pair[0]== '1' and question_only):
                train_q.append(encode_sent(vocab, qa_pair[2], 100))
                train_ra.append(encode_sent(vocab, qa_pair[3], 100))
                question_only = False
            if(qa_pair[0]=='0'and len(error_answer)< 10):
                error_answer.append(encode_sent(vocab, qa_pair[3], 100))
        if len(error_answer)<10:
            for i in range(0, 10-len(error_answer)):
                error_answer.append(padding)
        train_ea.append(error_answer)
    return np.array(train_q, dtype='float32'), np.array(train_ra, dtype='float32'), np.array(train_ea, dtype='float32')

#随机获取单个问题和答案以及答案的标注（是否最优答案）
def load_data2(trainList, vocab, batch_size, sequence_len):
    train_q, train_a, train_mark = [], [], []
    for i in range(0, batch_size):
        question_answer = trainList[random.randint(0, len(trainList) - 1)]
        train_q.append(encode_sent(vocab, question_answer[2], sequence_len))
        train_a.append(encode_sent(vocab, question_answer[3], sequence_len))
        train_mark.append(question_answer[0])
    return np.array(train_q, dtype='int32'), np.array(train_a, dtype='int32'), np.array(train_mark, dtype='int32')

#按顺序获取batch_size大小的问题答案对
def load_data3(trainList, vocab, batch_size, sequence_len, count, all_num):
    train_q, train_a, train_mark = [], [], []
    for i in range(0, batch_size):
        question_answer = trainList[(count[0]+i)%all_num]
        train_q.append(encode_sent(vocab, question_answer[2], sequence_len))
        train_a.append(encode_sent(vocab, question_answer[3], sequence_len))
        train_mark.append(question_answer[0])
    count[0] = (count[0] + batch_size)%all_num
    print 'count是：',count[0]
    return np.array(train_q, dtype='float32'), np.array(train_a, dtype='float32'), np.array(train_mark, dtype='int32')

#按顺序加载test集合的每一个问题和答案以及标注
def load_data_test(testList, vocab, batch_size, sequence_len):
    test_q, test_a, test_mark = [], [], []
    for i in range(0, batch_size):
        question_answer = testList[i]
        test_q.append(encode_sent(vocab, question_answer[2], sequence_len))
        test_a.append(encode_sent(vocab, question_answer[3], sequence_len))
        test_mark.append(question_answer[0])
    return np.array(test_q, dtype='float32'), np.array(test_a, dtype='float32'), np.array(test_mark, dtype='int32')

#获得测试集全部的问题分类和正确答案的位置，返回例如:[qid:0,19,3]表示第0个问题，有19个答案，第3个是正确答案
def get_test_seq(testList):
    qid = 'qid:0'
    test_seq = []
    seq_num = 0
    right_answer = 0
    all_num = 0
    for i in range(len(testList)):
        question_answer = testList[i]
        if(question_answer[1] == qid):
            seq_num += 1
            if(question_answer[0] == '1'):
                right_answer = i - all_num
        else:
            test_seq.append([qid, seq_num, right_answer])
            all_num += seq_num
            qid = question_answer[1]
            seq_num = 1
            right_answer = 0
            if (question_answer[0] == '1'):
                right_answer = i - all_num
    test_seq.append([qid, seq_num, right_answer])
    return test_seq



def build_qa_matrix(input_q, input_a, batch_size, sequece_len, embedding_size):
    a = T.matrix('a')
    b = T.matrix('b')
    c = T.dot(a, b)
    F_multiply = theano.function([a, b], c, allow_input_downcast=True)
    input_q = input_q.reshape((batch_size, sequece_len, embedding_size))
    input_a = input_a.reshape((batch_size, sequece_len, embedding_size))

    M1 = []
    for i in range(0, batch_size):
        word_to_word_qra = F_multiply(input_q[i], input_a[i].T)
        M1.append(word_to_word_qra)
    return np.array(M1, dtype='float32')

def validation(validate_model, testList, test_seq, vocab, batch_size, word_embeddings, sequence_len, embedding_size):
    test_len = len(testList)
    n = test_len/batch_size
    y = []
    probablity = []
    questiong_number = len(test_seq)
    accumulate = 0
    MRR_numbers = 0
    MRR_sum = 0.0
    for i in range(n):
        every_testList = testList[0+i*batch_size: batch_size+i*batch_size]
        test_q, test_a, test_mark = load_data_test(every_testList, vocab, batch_size, sequence_len)
        y.extend(test_mark)
        input_q = word_embeddings[np.array(test_q, dtype='int32')]
        input_a = word_embeddings[np.array(test_a, dtype='int32')]
        M1 = build_qa_matrix(input_q, input_a, batch_size, sequence_len, embedding_size)
        prediction_l, cost_l, error_l, probablity_l = validate_model(M1.reshape(batch_size, 1, sequence_len, embedding_size), test_mark)
        every_pro = probablity_l.T[1].tolist()
        probablity.extend(every_pro)
    for i in range(questiong_number):
        every_test_seq = test_seq[i]
        answer_numbers = every_test_seq[1]
        right_ans_num = every_test_seq[2]
        input_probablity = probablity[accumulate:accumulate+answer_numbers]
        accumulate += answer_numbers
        if(accumulate <= n*batch_size):
            MRR = answer_rank(input_probablity, right_ans_num, answer_numbers)
            MRR_sum += MRR
            MRR_numbers += 1
        else:break
    MRR_result = MRR_sum/MRR_numbers
    print 'MRR值为:',MRR_result
    with open(resultfile, 'a') as f:
        f.write('MRR: ' + str(MRR_result) + '\n')


#答案排序确定最好答案的位置
def answer_rank(probablity, right_ans, ans_numbers):
    rank = 1
    ritht_ans_probablity = probablity[right_ans]
    for i in range(ans_numbers):
        if(probablity[i] > ritht_ans_probablity):
            rank += 1
    MRR = 1.0/rank
    return MRR

class GRUQA(object):
    def __init__(self, M1, y, word_embeddings, batch_size, sequence_len, embedding_size, filter_size, num_filters, train_q, train_a):
        rng = np.random.RandomState(23455)
        self.params = []

        lookup_table = theano.shared(word_embeddings)
        self.params += [lookup_table]

        # 生成问题和答案的关联矩阵
        # input_q = word_embeddings[np.array(train_q, dtype='int32')]
        # input_a = word_embeddings[np.array(train_a, dtype='int32')]
        # M1 = build_qa_matrix(input_q, input_a, batch_size).reshape(batch_size, 1, sequence_len, embedding_size)
        # M1 = M1.reshape(batch_size, 1, sequence_len, embedding_size)

        U_M2 = np.random.uniform(-np.sqrt(1. / sequence_len), np.sqrt(1. / sequence_len), (6, sequence_len, sequence_len))
        W_M2 = np.random.uniform(-np.sqrt(1. / sequence_len), np.sqrt(1. / sequence_len), (6, sequence_len, sequence_len))
        b_M2 = np.zeros((6, sequence_len))
        UU_M2 = theano.shared(name='UU_M2', value=U_M2.astype(theano.config.floatX))
        WW_M2 = theano.shared(name='WW_M2', value=W_M2.astype(theano.config.floatX))
        bb_M2 = theano.shared(name='bb_M2',value=b_M2.astype(theano.config.floatX))
        self.UU_M2 = UU_M2
        self.WW_M2 = WW_M2
        self.bb_M2 = bb_M2

        def forward_prop_step(x_e, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_t = lookup_table[x_e, :]

            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(UU_M2[0].dot(x_t) + WW_M2[0].dot(s_t1_prev) + bb_M2[0])
            r_t1 = T.nnet.hard_sigmoid(UU_M2[1].dot(x_t) + WW_M2[1].dot(s_t1_prev) + bb_M2[1])
            c_t1 = T.tanh(UU_M2[2].dot(x_t) + WW_M2[2].dot(s_t1_prev * r_t1) + bb_M2[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(UU_M2[3].dot(s_t1) + WW_M2[3].dot(s_t2_prev) + bb_M2[3])
            r_t2 = T.nnet.hard_sigmoid(UU_M2[4].dot(s_t1) + WW_M2[4].dot(s_t2_prev) + bb_M2[4])
            c_t2 = T.tanh(UU_M2[5].dot(s_t1) + WW_M2[5].dot(s_t2_prev * r_t2) + bb_M2[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            return [s_t1, s_t2]

        x = T.ivector('x')
        [s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=-1,
            outputs_info=[dict(initial=T.zeros(sequence_len, dtype='float64')),
                          dict(initial=T.zeros(sequence_len, dtype='float64'))])
        result = s2
        ff = theano.function(inputs=[x], outputs=result, updates=updates)

        M2_input_q = []
        for i in range(batch_size):
            x1 = train_q[i]
            m = ff(x1)
            M2_input_q.append(m)
        M2_input_a = []
        for i in range(batch_size):
            x1 = train_a[i]
            m = ff(x1)
            M2_input_a.append(m)

        M2 = build_qa_matrix(np.array(M2_input_q), np.array(M2_input_a), batch_size, sequence_len,
                             embedding_size).reshape(batch_size, 1, sequence_len, embedding_size)

        # con2d_input = np.concatenate((M1, M2), axis=1)

        filter_shape = (num_filters, 2, filter_size, embedding_size)
        sentence_shape = (batch_size, 2, sequence_len, embedding_size)
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
        self.W = W
        self.b = b

        # 卷积+max_pooling
        conv_out = conv2d(input=M2, filters=W, filter_shape=filter_shape, input_shape=sentence_shape)
        # 卷积后的向量的长度为ds
        pooled_out = pool.pool_2d(input=conv_out, ws=(46, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

        #开始softmax
        input_softmax = T.reshape(pooled_active, (batch_size, num_filters))

        # #GRU处理向量
        # word_dim = 500
        # U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        # W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        # V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        # c = np.zeros((6, hidden_dim))
        # d = np.zeros(word_dim)

        softmax_shape = (num_filters, 2)
        W2 = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=softmax_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b2 = theano.shared(np.asarray(np.zeros(2), theano.config.floatX))
        self.W2 = W2
        self.b2 = b2

        probablity = T.nnet.softmax(T.dot(input_softmax, W2) + b2)
        prediction = T.argmax(probablity, 1)
        self.probablity = probablity
        self.prediction = prediction
        self.cost = -T.mean(T.log(probablity[T.arange(y.shape[0]), y]))
        self.error = T.mean(T.neq(prediction, y))


def train():
    num_filters = 500
    batch_size = 25
    filter_size = 5
    embedding_size = 50
    sequence_len = 50
    learning_rate = 0.01
    n_epochs = 20000
    validation_freq = 100

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    new_dict = {v:k for k,v in vocab.items()}
    trainList = load_train_list2()
    testList = load_test_list2()
    test_seq = get_test_seq(testList)
    train_q, train_a, train_mark = load_data2(trainList, vocab, batch_size, sequence_len)


    x1, x2 = T.tensor4('x1'), T.ivector('x2')
    model = GRUQA(M1=x1, y=x2, word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=train_q.shape[1], embedding_size=embedding_size,
                  filter_size=filter_size, num_filters=num_filters, train_q=train_q, train_a=train_a)

    cost = model.cost
    error = model.error
    prediction = model.prediction
    probablity = model.probablity
    print 'cost'
    print cost

    gw, gw2, gb, gb2, guum2, gwwm2, gbbm2 = T.grad(cost, [model.W, model.W2, model.b, model.b2, model.UU_M2, model.WW_M2, model.bb_M2])

    updates = [
        (model.W, model.W - learning_rate * gw),
        (model.W2, model.W2 - learning_rate * gw2),
        (model.b, model.b - learning_rate * gb),
        (model.b2, model.b2 - learning_rate * gb2),
        (model.UU_M2, model.UU_M2 - learning_rate * guum2),
        (model.WW_M2, model.WW_M2 - learning_rate * gwwm2),
        (model.bb_M2, model.bb_M2 - learning_rate * gbbm2)
        ]

    p1, p2 = T.tensor4('p1'), T.ivector('p2')

    train_model = theano.function(
        [p1, p2],
        [prediction, cost, error, probablity],
        updates=updates,
        givens={
            x1: p1, x2: p2
        }
    )

    v1, v2 = T.tensor4('v1'), T.ivector('v2')

    validate_model = theano.function(
        [v1, v2],
        [prediction, cost, error, probablity],
        givens={
            x1: v1, x2: v2
        }
    )

    epoch = 0
    done_looping = False
    print 'behind epoch=0'
    count = [0]
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        all_num = len(trainList)
        train_x1, train_x2, train_x3 = load_data3(trainList, vocab, batch_size, sequence_len, count, all_num)
        input_q = word_embeddings[np.array(train_x1, dtype='int32')]
        input_a = word_embeddings[np.array(train_x2, dtype='int32')]
        M1 = build_qa_matrix(input_q, input_a, batch_size, sequence_len, embedding_size)
        prediction_ij, cost_ij, error_ij, probablity_ij = train_model(M1.reshape(batch_size, 1, sequence_len, embedding_size), train_x3)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', error:' + str(error_ij)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            validation(validate_model, testList, test_seq, vocab, batch_size, word_embeddings, sequence_len, embedding_size)

if __name__ == '__main__':
    train()
    # num_filters = 500
    # batch_size = 5
    # filter_size = 5
    # embedding_size = 100
    # learning_rate = 0.02
    # n_epochs = 200
    # validation_freq = 100
    # sequence_len = 100
    #
    # vocab = build_vocab()
    # word_embeddings = load_word_embeddings(vocab, 100)
    # trainList = load_train_list2()
    # testList = load_test_list2()
    # test_seq = get_test_seq(testList)

    #测试MRR函数
    # test_len = len(testList)
    # n = test_len / batch_size
    # y = []
    # probablity = []
    # for i in range(n):
    #     every_testList = testList[0 + i * batch_size: batch_size + i * batch_size]
    #     test_q, test_a, test_mark = load_data_test(every_testList, vocab, batch_size)
    #     y.extend(test_mark)
    #     input_q = word_embeddings[np.array(test_q, dtype='int32')]
    #     input_a = word_embeddings[np.array(test_a, dtype='int32')]
    #     M1 = build_qa_matrix(input_q, input_a, batch_size)
    # probablity = [0.8,0.7,0.5,0.9,0.6]
    # MRR = answer_rank(probablity=probablity, right_ans=1, ans_numbers=5)
    # print MRR

    # train_q, train_a, train_mark = load_data2(trainList, vocab, batch_size, sequence_len)
    #
    # lookup_table = theano.shared(word_embeddings)
    # # input_q = lookup_table[T.cast(train_q.flatten(), dtype="int32")]
    # # input_a = lookup_table[T.cast(train_a.flatten(), dtype="int32")]
    # input_q = word_embeddings[np.array(train_q, dtype='int32')]
    # input_ra = word_embeddings[np.array(train_a, dtype='int32')]
    #
    # M1 = build_qa_matrix(input_q, input_q, batch_size, sequence_len, embedding_size).reshape(batch_size, 1, sequence_len, embedding_size)
    #
    # #GRU测试
    # hidden_dim = 100
    # U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
    # W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
    # bb = np.zeros((6, hidden_dim))
    # UU = theano.shared(name='U', value=U.astype(theano.config.floatX))
    # WW = theano.shared(name='W', value=W.astype(theano.config.floatX))
    #
    # def forward_prop_step(x_e, s_t1_prev, s_t2_prev):
    #     # This is how we calculated the hidden state in a simple RNN. No longer!
    #     # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
    #
    #     # Word embedding layer
    #     x_t = lookup_table[x_e,:]
    #
    #     # GRU Layer 1
    #     z_t1 = T.nnet.hard_sigmoid(UU[0].dot(x_t) + WW[0].dot(s_t1_prev) + bb[0])
    #     r_t1 = T.nnet.hard_sigmoid(UU[1].dot(x_t) + WW[1].dot(s_t1_prev) + bb[1])
    #     c_t1 = T.tanh(UU[2].dot(x_t) + WW[2].dot(s_t1_prev * r_t1) + bb[2])
    #     s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
    #
    #     # GRU Layer 2
    #     z_t2 = T.nnet.hard_sigmoid(UU[3].dot(s_t1) + WW[3].dot(s_t2_prev) + bb[3])
    #     r_t2 = T.nnet.hard_sigmoid(UU[4].dot(s_t1) + WW[4].dot(s_t2_prev) + bb[4])
    #     c_t2 = T.tanh(UU[5].dot(s_t1) + WW[5].dot(s_t2_prev * r_t2) + bb[5])
    #     s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
    #
    #     # Final output calculation
    #     # Theano's softmax returns a matrix with one row, we only need the row
    #     return [s_t1, s_t2]
    #
    #
    # x = T.ivector('x')
    # [s, s2], updates = theano.scan(
    #     forward_prop_step,
    #     sequences=x,
    #     truncate_gradient= -1 ,
    #     outputs_info=[dict(initial=T.zeros(hidden_dim,dtype='float64')),
    #                   dict(initial=T.zeros(hidden_dim,dtype='float64'))])
    # result = s2
    # ff = theano.function(inputs=[x], outputs=result , updates=updates)
    #
    # M2_input_q = []
    # for i in range(batch_size):
    #     x1 = train_q[i]
    #     m = ff(x1)
    #     M2_input_q.append(m)
    # M2_input_a = []
    # for i in range(batch_size):
    #     x1 = train_a[i]
    #     m = ff(x1)
    #     M2_input_a.append(m)
    #
    # M2 = build_qa_matrix(np.array(M2_input_q), np.array(M2_input_a), batch_size, sequence_len, embedding_size).reshape(batch_size, 1, sequence_len, embedding_size)
    #组合word_to_word矩阵和sentence_to_sentence矩阵
    # con2d_input = np.concatenate((M1,M2),axis=1)
    # rng = np.random.RandomState(23455)
    # filter_shape = (num_filters, 1, 5, 100)
    # sentence_shape = (batch_size, 1, 100, 100)
    # fan_in = np.prod(filter_shape[1:])
    # fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    # W_bound = np.sqrt(6. / (fan_in + fan_out))
    # W = theano.shared(
    #     np.asarray(
    #         rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
    #         dtype=theano.config.floatX
    #     ),
    #     borrow=True
    # )
    # b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
    # b = theano.shared(value=b_values, borrow=True)
    #
    # outputs_1, outputs_2, outputs_3 = [], [], []
    # # 卷积+max_pooling
    # conv_out = conv2d(input=M1, filters=W, filter_shape=filter_shape, input_shape=sentence_shape)
    # # 卷积后的向量的长度为ds
    # pooled_out = pool.pool_2d(input=conv_out, ws=(96, 1), ignore_border=True, mode='max')
    # pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
    # # pooled_active.reshape(batch_size, num_filters)
    # outputs_1.append(pooled_active)
    #
    # input_softmax = T.reshape(pooled_active,(batch_size ,num_filters))
    # softmax_shape = (num_filters, 2)
    # W2 = theano.shared(
    #     np.asarray(
    #         rng.uniform(low=-W_bound, high=W_bound, size=softmax_shape),
    #         dtype=theano.config.floatX
    #     ),
    #     borrow=True
    # )
    # y = T.ivector('y')
    # b = theano.shared(np.asarray(np.zeros(2), theano.config.floatX))
    # probalblity = T.nnet.softmax(T.dot(input_softmax, W2) + b)
    # output = y
    # f = theano.function([y],output)
    # y_softmax = f(train_mark)
    # cost = -T.mean(T.log(probalblity[T.arange(y.shape[0]), y]))
    # prediction = T.argmax(probalblity, 1)
    # print 't'





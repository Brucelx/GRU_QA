#-*-coding:utf-8-*-
#!/usr/bin/env python

import numpy as np
import sys
import os, sys, timeit, random, operator

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import pickle

reload(sys)
sys.setdefaultencoding( "utf-8" )
# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.exception_verbosity= 'high' # Use 'warn' to activate this feature
theano.config.optimizer= 'fast_compile'
theano.config.floatX = 'float32'

trainfile = unicode('E:/git/GRU_QA/GRU_QA/nlpcc_train','utf-8')
testfile = unicode('E:/git/GRU_QA/GRU_QA/nlpcc_test','utf-8')
vectorsfile = unicode('E:/git/GRU_QA/GRU_QA/vectors.bin','utf-8')
resultfile = unicode('E:/git/GRU_QA/GRU_QA/output.txt','utf-8')
paramsfile = unicode('E:/git/GRU_QA/GRU_QA/params.txt','utf-8')
trainfile_linux = unicode('/home/liuxiao/GRU/nlpcc_train','utf-8')
testfile_linux = unicode('/home/liuxiao/GRU/nlpcc_test','utf-8')
vectorsfile_linux = unicode('/home/liuxiao/GRU/vectors_100.bin','utf-8')
resultfile_linux = unicode('/home/liuxiao/GRU/output.txt','utf-8')
paramsfile_linux = unicode('/home/liuxiao/GRU/params.txt','utf-8')

load_params_file = unicode('/home/liuxiao/GRU//params.txt.npz','utf-8')

if(os.path.exists('/home/liuxiao/GRU/nlpcc_train')):
    trainfile = trainfile_linux
    testfile = testfile_linux
    vectorsfile = vectorsfile_linux
    resultfile = resultfile_linux
    paramsfile = paramsfile_linux

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
    x, m = [], []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
        if words[i] == '<a>':  # TODO
            m.append(1)  # fixed sequence length, else use 0
        else:
            m.append(1)
    return x, m

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
    mask_1, mask_2 = [], []
    for i in range(0, batch_size):
        question_answer = trainList[random.randint(0, len(trainList) - 1)]
        x, m = encode_sent(vocab, question_answer[2], sequence_len)
        train_q.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, question_answer[3], sequence_len)
        train_a.append(x)
        mask_2.append(m)
        train_mark.append(question_answer[0])
    return np.transpose(np.array(train_q, dtype=theano.config.floatX)), np.transpose(np.array(train_a, dtype=theano.config.floatX)), np.array(train_mark, dtype='int32'),\
           np.transpose(np.array(mask_1, dtype=theano.config.floatX)), np.transpose(np.array(mask_2, dtype=theano.config.floatX))

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
    mask_1, mask_2 = [], []
    for i in range(0, batch_size):
        question_answer = testList[i]
        x, m = encode_sent(vocab, question_answer[2], sequence_len)
        test_q.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, question_answer[3], sequence_len)
        test_a.append(x)
        mask_2.append(m)
        test_mark.append(question_answer[0])
    return np.transpose(np.array(test_q, dtype=theano.config.floatX)), np.transpose(np.array(test_a, dtype=theano.config.floatX)), np.array(test_mark, dtype='int32'), \
           np.transpose(np.array(mask_1, dtype=theano.config.floatX)), np.transpose(
        np.array(mask_2, dtype=theano.config.floatX))

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
        test_q, test_a, test_mark, mask1, mask2 = load_data_test(every_testList, vocab, batch_size, sequence_len)
        y.extend(test_mark)
        prediction_l, cost_l, error_l, probablity_l = validate_model( test_q, test_a, test_mark, mask1, mask2)
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

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def gener_matrix(a, b, batch_size):
    result = []
    for i in range(batch_size):
        mm = T.dot(a[i], b[i])
        result.append(mm)
    return result

def param_init_lstm(proj_size, tparams, grad_params):
    W = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    W_t = theano.shared(W, borrow=True)
    tparams[_p('lstm', 'W')] = W_t
    U = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    U_t = theano.shared(U, borrow=True)
    tparams[_p('lstm', 'U')] = U_t
    b = np.zeros((4 * proj_size,))
    b_t = theano.shared(b.astype(theano.config.floatX), borrow=True)
    tparams[_p('lstm', 'b')] = b_t
    grad_params += [W_t, U_t, b_t]

    return tparams, grad_params

def param_init_cnn(filter_sizes, num_filters, proj_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
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
        tparams['cnn_W_' + str(filter_size)] = W
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        tparams['cnn_b_' + str(filter_size)] = b
        grad_params += [W, b]
    return tparams, grad_params

def param_init_softmax(num_filters_total, embedding_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    softmax_shape = (num_filters_total, 2)
    filter_shape = (num_filters_total, 1, 1, embedding_size)
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W2 = theano.shared(
        np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=softmax_shape),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    tparams[_p('softmax', 'W')] = W2
    b_values = np.asarray(np.zeros(2), theano.config.floatX)
    b2 = theano.shared(value=b_values, borrow=True)
    tparams[_p('softmax', 'b')] = b2
    grad_params += [W2, b2]
    return tparams, grad_params

class QAMODEL(object):
    def __init__(self,word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters):
        self.params, tparams = [], {}
        tparams, self.params = param_init_lstm(embedding_size, tparams, self.params)
        tparams, self.params = param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, self.params)
        tparams, self.params = param_init_softmax(len(filter_sizes)*num_filters, embedding_size, tparams, self.params)

        train_q, train_a, y = T.fmatrix('train_q'), T.fmatrix('train_a'), T.ivector('y')
        mask1, mask2 = T.fmatrix('mask1'), T.fmatrix('mask2')

        lookup_table = theano.shared(word_embeddings, borrow=True)
        tparams['lookup_table'] = lookup_table
        self.params += [lookup_table]
        self.tparams = tparams

        n_timesteps = train_q.shape[0]
        n_samples = train_q.shape[1]

        lstm1, lstm_whole1 = self._lstm_net(tparams, train_q, sequence_len, batch_size, embedding_size, mask1)
        lstm2, lstm_whole2 = self._lstm_net(tparams, train_a, sequence_len, batch_size, embedding_size, mask2)

        #生成cnn双通道输入，分别是字关联矩阵和词关联矩阵
        word_related_matrix = self._word_related_matrix(tparams, train_q, train_a, sequence_len, batch_size, embedding_size)
        # phrase_related_matrix = T.reshape(gener_matrix(lstm1.dimshuffle(1,0,2),lstm2.dimshuffle(1,2,0),batch_size),[batch_size, 1, sequence_len, embedding_size])
        # cnn_two_pipe = T.concatenate([word_related_matrix, phrase_related_matrix], axis=1)

        #cnn单通道输入，词关联矩阵
        related_matrix = gener_matrix(lstm1.dimshuffle(1, 0, 2), lstm2.dimshuffle(1, 2, 0), batch_size)
        cnn_input = T.reshape(related_matrix, [batch_size, 1, sequence_len, embedding_size])

        cnn_result = self._cnn_net(tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)

        probablity = T.nnet.softmax(T.dot(cnn_result, tparams['softmax_W']) + tparams['softmax_b'])
        prediction = T.argmax(probablity, 1)
        self.probablity = probablity
        self.prediction = prediction
        self.cost = -T.mean(T.log(probablity[T.arange(y.shape[0]), y]))
        self.error = T.mean(T.neq(prediction, y))

    ##生成字关联矩阵
    def _word_related_matrix(self,tparams, train_q, train_a,sequence_len, batch_size, embedding_size,):
        train_q = tparams['lookup_table'][T.cast(train_q.dimshuffle(1,0).flatten(), dtype="int32")]
        train_a = tparams['lookup_table'][T.cast(train_a.dimshuffle(1,0).flatten(), dtype="int32")]
        word_matrix_q = train_q.reshape((batch_size,sequence_len, embedding_size))
        word_matrix_a = train_a.reshape((batch_size, sequence_len, embedding_size)).dimshuffle(0,2,1)
        word_related_matrix = T.reshape(gener_matrix(word_matrix_q, word_matrix_a, batch_size),(batch_size,1,sequence_len,embedding_size))
        return word_related_matrix

    def _cnn_net(self, tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, embedding_size):
        outputs = []
        for filter_size in filter_sizes:
            filter_shape = (num_filters, 1, filter_size, embedding_size)
            input_shape = (batch_size, 1, sequence_len, embedding_size)
            W = tparams['cnn_W_' + str(filter_size)]
            b = tparams['cnn_b_' + str(filter_size)]
            conv_out = conv2d(input=cnn_input, filters=W, filter_shape=filter_shape, input_shape=input_shape)
            pooled_out = pool.pool_2d(input=conv_out, ws=(sequence_len - filter_size + 1, 1), ignore_border=True,
                                      mode='max')
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
            outputs.append(pooled_active)
        num_filters_total = num_filters * len(filter_sizes)
        output_tensor = T.reshape(T.concatenate(outputs, axis=1), [batch_size, num_filters_total])
        return output_tensor

    def _lstm_net(self, tparams, _input, sequence_len, batch_size, embedding_size, mask):
        input_matrix = tparams['lookup_table'][T.cast(_input.flatten(), dtype="int32")]
        input_x = input_matrix.reshape((sequence_len, batch_size, embedding_size))
        proj, proj_whole = lstm_layer(tparams, input_x, embedding_size, prefix='lstm', mask=mask)
        # if useMask == True:
        # proj = (proj * mask[:, :, None]).sum(axis=0)
        # proj = proj / mask.sum(axis=0)[:, None]
        # if options['use_dropout']:
        # proj = dropout_layer(proj, use_noise, trng)
        return proj, proj_whole

def lstm_layer(tparams, state_below, embedding_size, prefix='lstm', mask=None):
    #dim-0 steps, dim-1 samples(batch_size), dim-3 word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #h means hidden output? c means context? so we'll use h?
    #rval[0] = [sequence_len, batch_size, proj_size], rval[1] the same

    #so preact size must equl to x_(lstm input slice)
    #if you want change lstm h(t) size, 'lstm_U' and 'lstm_b'
    #and precat must be changed to another function, like h*U+b
    #see http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #f(t) = sigmoid(Wf * [h(t-1),x(t)] + bf)
    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, embedding_size))
        f = T.nnet.sigmoid(_slice(preact, 1, embedding_size))
        o = T.nnet.sigmoid(_slice(preact, 2, embedding_size))
        c = T.tanh(_slice(preact, 3, embedding_size))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        #if mask(t-1)==0, than make h(t) = h(t-1)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = embedding_size
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0], rval[1]

def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
        tparams=model.tparams)
    print "Saved model parameters to %s." % outfile

# def load_model_parameters(path, modelClass=QAMODEL):
#     npzfile = np.load(path)
#     tparams = npzfile["tparams"]
#     print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
#     sys.stdout.flush()
#     model = modelClass(word_dim, hidden_dim=hidden_dim)
#     model.tparams = tparams
#     return model

def train():
    num_filters = 500
    batch_size = 256
    filter_sizes = [1,2,3,5]
    embedding_size = 100
    sequence_len = 100
    learning_rate = 0.01
    n_epochs = 2000000
    validation_freq = 1000

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    new_dict = {v:k for k,v in vocab.items()}
    trainList = load_train_list2()
    testList = load_test_list2()
    test_seq = get_test_seq(testList)

    x1, x2, x3 = T.fmatrix('x1'), T.fmatrix('x2'), T.ivector('x3')
    m1, m2  = T.fmatrix('m1'), T.fmatrix('m2')
    model = QAMODEL(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len, embedding_size=embedding_size,
                  filter_sizes=filter_sizes, num_filters=num_filters)

    cost = model.cost
    error = model.error
    prediction = model.prediction
    probablity = model.probablity
    params = model.params
    print 'cost'
    print cost

    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    p1, p2, p3 = T.fmatrix('p1'), T.fmatrix('p2'), T.ivector('p3')
    q1, q2 = T.fmatrix('q1'), T.fmatrix('q2')
    train_model = theano.function(
        [p1, p2, p3, q1, q2],
        [prediction, cost, error, probablity],
        updates=updates,
        givens={
            x1: p1, x2: p2, x3: p3, m1: q1, m2: q2
        }
    )

    v1, v2, v3 = T.fmatrix('v1'), T.fmatrix('v2'), T.ivector('v3')
    u1, u2 = T.fmatrix('u1'), T.fmatrix('u2')

    validate_model = theano.function(
        [v1, v2, v3, u1, u2],
        [prediction, cost, error, probablity],
        givens={
            x1: v1, x2: v2, x3: v3, m1:u1, m2:u2
        }
    )

    epoch = 0
    done_looping = False
    print 'behind epoch=0'
    count = [0]
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_q, train_a, train_mark, mask1, mask2 = load_data2(trainList, vocab, batch_size, sequence_len)
        prediction_ij, cost_ij, error_ij, probablity_ij = train_model(p1 = train_q, p2 = train_a, p3 = train_mark, q1=mask1, q2=mask2)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', error:' + str(error_ij)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            validation(validate_model, testList, test_seq, vocab, batch_size, word_embeddings, sequence_len, embedding_size)
            save_model_parameters_theano(model, paramsfile)

def predict():
    load_model_parameters_theano(paramsfile,)

if __name__ == '__main__':
    train()
    # num_filters = 500
    # batch_size = 5
    # filter_sizes = [1,2,5]
    # embedding_size = 100
    # learning_rate = 0.01
    # n_epochs = 200
    # validation_freq = 100
    # sequence_len = 100
    #
    # vocab = build_vocab()
    # word_embeddings = load_word_embeddings(vocab, 100)
    # trainList = load_train_list2()
    # testList = load_test_list2()
    # test_seq = get_test_seq(testList)
    #
    # train_q, train_a, train_mark, mask1, mask2 = load_data2(trainList, vocab, batch_size, sequence_len)
    #
    # params, tparams = [], {}
    # tparams, params = param_init_lstm(embedding_size, tparams, params)
    # tparams, params = param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, params)
    # tparams, params = param_init_softmax(len(filter_sizes) * num_filters, embedding_size, tparams, params)
    #
    # lookup_table = theano.shared(word_embeddings, borrow=True)
    # tparams['lookup_table'] = lookup_table
    # params += [lookup_table]
    #
    # lstm1, lstm_whole1 = lstm_net(tparams, train_q, sequence_len, batch_size, embedding_size, mask1,
    #                                     embedding_size)
    # lstm2, lstm_whole2 = lstm_net(tparams, train_a, sequence_len, batch_size, embedding_size, mask2,
    #                                     embedding_size)
    #
    # related_matrix = gener_matrix(lstm1.dimshuffle(1, 0, 2), lstm2.dimshuffle(1, 2, 0), batch_size)
    # cnn_input = T.reshape(related_matrix, [batch_size, 1, sequence_len, embedding_size])
    #
    # print 't'

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






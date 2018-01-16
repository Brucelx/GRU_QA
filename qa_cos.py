#-*-coding:utf-8-*-
#!/usr/bin/env python

import numpy as np
import sys
import os, sys, timeit, random, operator

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import cPickle
from gensim.models import KeyedVectors

reload(sys)
sys.setdefaultencoding( "utf-8" )
# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.exception_verbosity= 'high' # Use 'warn' to activate this feature
theano.config.optimizer= 'fast_compile'
theano.config.floatX = 'float32'

trainfile = unicode('E:/git/GRU_QA/corpus/nlpcc_train1','utf-8')
train0file = unicode('E:/git/GRU_QA/corpus/nlpcc_train0','utf-8')
testfile = unicode('E:/git/GRU_QA/corpus/nlpcc_test','utf-8')
vectorsfile = unicode('E:/git/GRU_QA/corpus/vectors_100.bin','utf-8')
outputfile = unicode('E:/git/GRU_QA/corpus/output.txt','utf-8')
paramsfile = unicode('E:/git/GRU_QA/corpus/params.txt','utf-8')

# trainfile_linux = unicode('/home/liuxiao/tongji/train1_data_version_1','utf-8')
# train0file_linux = unicode('/home/liuxiao/tongji/train0_data_version_1','utf-8')
# testfile_linux = unicode('/home/liuxiao/tongji/test_data_version_1','utf-8')
# vectorsfile_linux = unicode('/home/liuxiao/tongji/vectors_100.bin','utf-8')
# outputfile_linux = unicode('/home/liuxiao/GRU/cnn2/output.txt','utf-8')
# paramsfile_linux = unicode('/home/liuxiao/GRU/cnn2/params.txt','utf-8')

trainfile_linux = unicode('/home/liuxiao/tongji/train1_data_version_1','utf-8')
train0file_linux = unicode('/home/liuxiao/tongji/train0_data_version_1','utf-8')
testfile_linux = unicode('/home/liuxiao/tongji/test_data_version_1','utf-8')
vectorsfile_linux = unicode('/home/liuxiao/tongji/vectors_300.bin','utf-8')
outputfile_linux = unicode('/home/liuxiao/GRU/cnn1/output.txt','utf-8')
paramsfile_linux = unicode('/home/liuxiao/GRU/cnn1/params.txt','utf-8')

validationfile = unicode('/home/liuxiao/tongji/nlpcc_validation','utf-8')


if(os.path.exists('/home/liuxiao/GRU/nlpcc_validation')):
    trainfile = trainfile_linux
    train0file = train0file_linux
    testfile = testfile_linux
    vectorsfile = vectorsfile_linux
    outputfile = outputfile_linux
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

# def load_vectors(embedding_size):
#     global vectorsfile
#     vectors = {}
#     for line in open(vectorsfile):
#         items = line.strip().split(' ')
#         if len(items[0]) <= 0:
#             continue
#         vec = []
#         for i in range(1, embedding_size+1):
#             vec.append(float(items[i]))
#         vectors[items[0]] = vec
#     return vectors
#
# def load_word_embeddings(vocab, embedding_size):
#     vectors = load_vectors(embedding_size)
#     embeddings = [] #brute initialization
#     for i in range(0, len(vocab)):
#         vec = []
#         for j in range(0, embedding_size):
#             vec.append(0.01)
#         embeddings.append(vec)
#     for word, code in vocab.items():
#         if word in vectors:
#             embeddings[code] = vectors[word]
#     return np.array(embeddings, dtype='float32')

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
    for line in open(trainfile):
        trainList.append(line.strip().split('\t'))
    return trainList

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

def load_test_list(testfile):
    testList = []
    for line in open(testfile):
        testList.append(line.strip().split('\t'))
    return testList

def check_int(id):
    id = str(id)
    for i in id:
        if i < '0' or i > '9':
            return False
    return True

def load_train_data_from_2files(train0Dict, train1List, vocab, batch_size, words_num_dim):
    train_1, train_2, train_3 = [], [], []
    mask = []
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

        x, m = encode_sent(vocab, pos[2], words_num_dim)
        train_1.append(x)
        x, m = encode_sent(vocab, pos[3], words_num_dim)
        train_2.append(x)
        x, m = encode_sent(vocab, neg, words_num_dim)
        train_3.append(x)
        mask.append(m)
        cnt += 1
        if cnt >= batch_size:
            break
    return np.transpose(np.array(train_1, dtype=theano.config.floatX)), np.transpose(np.array(train_2, dtype=theano.config.floatX)), \
           np.transpose(np.array(train_3, dtype=theano.config.floatX)), np.transpose(np.array(mask, dtype=theano.config.floatX))


#按顺序加载test集合的每一个问题和答案以及标注
def load_test_data(testList, vocab, index, batch_size, sequence_len):
    x1, x2, x3 = [], [], []
    mask = []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        question_answer = testList[true_index]
        x, m = encode_sent(vocab, question_answer[2], sequence_len)
        x1.append(x)
        mask.append(m)
        x, m = encode_sent(vocab, question_answer[3], sequence_len)
        x2.append(x)
        x3.append(x)
    return np.transpose(np.array(x1, dtype=theano.config.floatX)), np.transpose(np.array(x2, dtype=theano.config.floatX)), \
           np.transpose(np.array(x3, dtype=theano.config.floatX)), np.transpose(np.array(mask, dtype=theano.config.floatX))

def validation(validate_model, testList, vocab, batch_size, words_num_dim):
    index, score_list = int(0), []
    while True:
        x1, x2, x3, mask = load_test_data(testList, vocab, index, batch_size, words_num_dim)
        batch_scores, nouse = validate_model(x1, x2, x3, mask)
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
    map_sum = float(0)
    mrr_sum = float(0)
    #qid_count = 0

    for qid, items in sdict.items():
        items.sort(key=operator.itemgetter(0), reverse=True)
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
            map_index_down += 1
            if flag == '1':
                map_index_up += 1
                temp_map_sum += float(map_index_up) / float(map_index_down)
        temp_map_sum /= float(map_index_up)
        map_sum += temp_map_sum
    mrr_sum /= float(qa_count)
    map_sum /= float(qa_count)
    print 'MRR值为:', str(mrr_sum), ' MAP值为:', str(map_sum)
    with open(outputfile, 'a') as f:
        f.write('MRR: ' + str(mrr_sum) + ' MAP值为: ' + str(map_sum) + '\n')
    return mrr_sum

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)


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

def param_init_leaner_transform(embedding_size, tparams, grad_params):
    W = np.asarray(ortho_weight(embedding_size))
    W_lt = theano.shared(W, borrow=True)
    tparams[_p('leaner_transform', 'W')] = W_lt
    grad_params += [W_lt]

    return tparams, grad_params

class QAMODEL(object):
    def __init__(self,word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters, learning_rate, margin_value):
        self.params, tparams = [], {}
        tparams, self.params = param_init_lstm(embedding_size, tparams, self.params)
        tparams, self.params = param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, self.params)
        # tparams, self.params = param_init_softmax(len(filter_sizes)*num_filters, embedding_size, tparams, self.params)
        # tparams, self.params = param_init_leaner_transform(embedding_size, tparams, self.params)

        train_1, train_2, train_3 = T.fmatrix('train_1'), T.fmatrix('train_2'), T.fmatrix('train_3')
        mask= T.fmatrix('mask')

        lookup_table = theano.shared(word_embeddings, borrow=True)
        tparams['lookup_table'] = lookup_table
        self.params += [lookup_table]
        self.tparams = tparams
        params = self.params

        lstm1, lstm_whole1 = self._lstm_net(tparams, train_1, sequence_len, batch_size, embedding_size, mask)
        lstm2, lstm_whole2 = self._lstm_net(tparams, train_2, sequence_len, batch_size, embedding_size, mask)
        lstm3, lstm_whole3 = self._lstm_net(tparams, train_3, sequence_len, batch_size, embedding_size, mask)

        cnn_input_ans = T.reshape(lstm1.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, embedding_size])
        cnn_input_pos = T.reshape(lstm2.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, embedding_size])
        cnn_input_neg = T.reshape(lstm3.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, embedding_size])

        cnn1 = self._cnn_net(tparams, cnn_input_ans, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
        cnn2 = self._cnn_net(tparams, cnn_input_pos, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
        cnn3 = self._cnn_net(tparams, cnn_input_neg, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)

        len1 = T.sqrt(T.sum(cnn1 * cnn1, axis=1))
        len2 = T.sqrt(T.sum(cnn2 * cnn2, axis=1))
        len3 = T.sqrt(T.sum(cnn3 * cnn3, axis=1))

        self.cos12 = T.sum(cnn1 * cnn2, axis=1) / (len1 * len2)
        self.cos13 = T.sum(cnn1 * cnn3, axis=1) / (len1 * len3)

        zero = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)
        margin = theano.shared(np.full(batch_size, margin_value, dtype=theano.config.floatX), borrow=True)
        diff = T.cast(T.maximum(zero, margin - self.cos12 + self.cos13), dtype=theano.config.floatX)
        self.cost = T.sum(diff, acc_dtype=theano.config.floatX)
        self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)


        grads = T.grad(self.cost, params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [train_1, train_2, train_3, mask],
            [self.cost, self.accuracy],
            updates=updates
        )

        self.validate_model = theano.function(
            [train_1, train_2, train_3, mask],
            [self.cos12, self.cos13]
        )


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

def save_model_parameters_theano(model, outfile, filter_sizes):
    save_file = open(outfile, 'wb')
    tparams = model.tparams
    cPickle.dump(tparams[_p('lstm', 'W')].get_value(borrow=True), save_file, -1)
    cPickle.dump(tparams[_p('lstm', 'U')].get_value(borrow=True), save_file, -1)
    cPickle.dump(tparams[_p('lstm', 'b')].get_value(borrow=True), save_file, -1)

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
    tparams[_p('lstm', 'W')].set_value(cPickle.load(save_file), borrow=True)
    tparams[_p('lstm', 'U')].set_value(cPickle.load(save_file), borrow=True)
    tparams[_p('lstm', 'b')].set_value(cPickle.load(save_file), borrow=True)

    for filter_size in filter_sizes:
        tparams[_p('cnn_W', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)
        tparams[_p('cnn_b', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)

    tparams[_p('lookup','table')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('softmax', 'W')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('softmax', 'b')].set_value(cPickle.load(save_file), borrow=True)

    # tparams[_p('leaner_transform', 'W')].set_value(cPickle.load(save_file), borrow=True)

    print "Building model model from %s " % (path)
    save_file.close()

def train():
    num_filters = 500
    batch_size = 256
    filter_sizes = [1,2,3]
    embedding_size = 300
    sequence_len = 50
    learning_rate = 0.001
    n_epochs = 2000000
    validation_freq = 100
    margin_value = 0.05

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    train0Dict = load_train0_dict()
    testList = load_test_list(testfile)

    model = QAMODEL(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len, embedding_size=embedding_size,
                  filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=learning_rate, margin_value=margin_value)

    epoch = 0
    done_looping = False
    print 'behind epoch=0'
    best_MRR = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_x1, train_x2, train_x3, mask = load_train_data_from_2files(train0Dict, trainList, vocab, batch_size,sequence_len)
        cost, accuracy = model.train_model(train_x1, train_x2, train_x3, mask)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost) + ', accuracy:' + str(accuracy)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            MRR_result = validation(model.validate_model, testList, vocab, batch_size, sequence_len)
            if(MRR_result > best_MRR):
                save_model_parameters_theano(model, paramsfile, filter_sizes)
                best_MRR = MRR_result
                print 'best_MRR .............',best_MRR

def predict(modelClass=QAMODEL):
    num_filters = 500
    batch_size = 256
    filter_sizes = [1, 2, 3]
    embedding_size = 300
    sequence_len = 50
    learning_rate = 0.001
    margin_value = 0.05

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    model = modelClass(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len,
                       embedding_size=embedding_size,
                       filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=learning_rate, margin_value=margin_value)

    load_model_parameters(model, paramsfile, filter_sizes)

    print 'start'
    testList = load_test_list(validationfile)
    print 'Evaluation ......'
    validation(model.validate_model, testList, vocab, batch_size, sequence_len)
    print 'end'


if __name__ == '__main__':
    train()
    # predict()
    # num_filters = 50
    # batch_size = 10
    # filter_sizes = [1]
    # embedding_size = 100
    # sequence_len = 50
    # learning_rate = 0.005
    # n_epochs = 2000000
    # validation_freq = 100
    # margin_value = 0.05
    #
    # vocab = build_vocab()
    # word_embeddings = load_word_embeddings(vocab, embedding_size)
    # trainList = load_train_list()
    # train0Dict = load_train0_dict()
    # testList = load_test_list(testfile)
    #
    # model = QAMODEL(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len,
    #                 embedding_size=embedding_size,
    #                 filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=learning_rate,
    #                 margin_value=margin_value)
    # train_x1, train_x2, train_x3, mask = load_train_data_from_2files(train0Dict, trainList, vocab, batch_size,
    #                                                                  sequence_len)
    # train_x1 = np.transpose(train_x1)
    # train_x2 = np.transpose(train_x2)
    # train_x3 = np.transpose(train_x3)
    #
    # params, tparams = [], {}
    # tparams, params = param_init_lstm(embedding_size, tparams, params)
    # tparams, params = param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, params)
    #
    # lookup_table = theano.shared(word_embeddings, borrow=True)
    # tparams['lookup_table'] = lookup_table
    #
    # input_matrix1 = lookup_table[T.cast(train_x1.flatten(), dtype="int32")]
    # input_matrix2 = lookup_table[T.cast(train_x2.flatten(), dtype="int32")]
    # input_matrix3 = lookup_table[T.cast(train_x3.flatten(), dtype="int32")]
    #
    # # CNN的输入是4维矩阵，这里只是增加了一个维度而已
    # input_x1 = input_matrix1.reshape((batch_size, 1, sequence_len, embedding_size))
    # input_x2 = input_matrix2.reshape((batch_size, 1, sequence_len, embedding_size))
    # input_x3 = input_matrix3.reshape((batch_size, 1, sequence_len, embedding_size))
    #
    # cnn1 = model._cnn_net(tparams, input_x1, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
    # cnn2 = model._cnn_net(tparams, input_x2, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
    # cnn3 = model._cnn_net(tparams, input_x3, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
    #
    # len1 = T.sqrt(T.sum(cnn1 * cnn1, axis=1))
    # len2 = T.sqrt(T.sum(cnn2 * cnn2, axis=1))
    # len3 = T.sqrt(T.sum(cnn3 * cnn3, axis=1))
    #
    # cos12 = T.sum(cnn1 * cnn2, axis=1) / (len1 * len2)
    # cos13 = T.sum(cnn1 * cnn3, axis=1) / (len1 * len3)
    #
    # zero = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)
    # margin = theano.shared(np.full(batch_size, margin_value, dtype=theano.config.floatX), borrow=True)
    # diff = T.cast(T.maximum(zero, margin - cos12 + cos13), dtype=theano.config.floatX)
    # cost = T.sum(diff, acc_dtype=theano.config.floatX)
    # accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)
    # print 't'

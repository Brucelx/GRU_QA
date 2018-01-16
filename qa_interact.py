#-*-coding:utf-8-*-
#!/usr/bin/env python

import numpy as np
import sys
import os, sys, timeit, random, operator, time

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import cPickle
from gensim.models import KeyedVectors


from logUtil import log
from timeUtil import GetNowTime


reload(sys)
sys.setdefaultencoding( "utf-8" )
# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.exception_verbosity= 'high' # Use 'warn' to activate this feature
theano.config.optimizer= 'fast_compile'
theano.config.floatX = 'float32'

version_num = '1'
trainfile = '/home/liuxiao/tongji/train1_data_version_' + version_num
train0file = '/home/liuxiao/tongji/train0_data_version_' + version_num
test1file = '/home/liuxiao/tongji/test_data_version_' + version_num
vectorsfile = '/home/liuxiao/tongji/vectors_300.bin'
paramsfile = '/home/liuxiao/tongji/test/parameters'
validationfile = '/home/liuxiao/tongji/nlpcc_validation'
#vectorsfile = '/opt/exp_data/word_vector_cn/all_dbqa.bin'

#log file
logfile_path = '/home/liuxiao/tongji/test'

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
    print 'trainfile:',len(vocab)
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
    print 'train0file:', len(vocab)
    return vocab

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
    global train0file
    trainList = []
    for line in open(trainfile):
        trainList.append(line.strip().split('\t'))
    for line in open(train0file):
        trainList.append(line.strip().split('\t'))
    return trainList

def load_test_list(testfile):
    testList = []
    for line in open(testfile):
        testList.append(line.strip().split('\t'))
    return testList


#按顺序加载test集合的每一个问题和答案以及标注
def load_test_data(testList, vocab, index, batch_size, sequence_len):
    test_q, test_a, test_mark = [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        question_answer = testList[true_index]
        x = encode_sent(vocab, question_answer[2], sequence_len)
        test_q.append(x)
        x = encode_sent(vocab, question_answer[3], sequence_len)
        test_a.append(x)
        mark = question_answer[0].strip().split("_")[-1]
        test_mark.append(mark)
    return np.array(test_q, dtype=theano.config.floatX), np.array(test_a, dtype=theano.config.floatX), np.array(test_mark, dtype='int32')

#随机获取单个问题和答案以及答案的标注（是否最优答案）
def load_train_data(trainList, vocab, batch_size, sequence_len):
    train_q, train_a, train_mark = [], [], []
    for i in range(0, batch_size):
        question_answer = trainList[random.randint(0, len(trainList) - 1)]
        x = encode_sent(vocab, question_answer[2], sequence_len)
        train_q.append(x)
        x = encode_sent(vocab, question_answer[3], sequence_len)
        train_a.append(x)
        mark = question_answer[0].strip().split("_")[-1]
        train_mark.append(mark)
    return np.array(train_q, dtype=theano.config.floatX), np.array(train_a, dtype=theano.config.floatX), np.array(train_mark, dtype='int32')


def validation(validate_model, testList, vocab, batch_size, words_num_dim):
    index, score_list = int(0), []
    while True:
        x1, x2, x3 = load_test_data(testList, vocab, index, batch_size, words_num_dim)
        wo_q1, wo_a1 = gener_wordoverlap(x1, x2, batch_size, words_num_dim, vocab)
        batch_scores, nouse = validate_model(x1, x2, x3, 1.0, wo_q1, wo_a1)
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


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def _p(pp, name):
    return '%s_%s' % (pp, name)

def gener_wordoverlap(ques_matrix, ans_matrix, batch_size, sentence_len, vocab):
    ques_wordoverlap = []
    ans_wordoverlap = []
    for i in range(batch_size):
        ques_each_wo = []
        ans_each_wo = []
        for j in range(sentence_len):
            similarity_ques = 0.01
            similarity_ans = 0.01
            for k in range(sentence_len):
                if(ques_matrix[i][j] == ans_matrix[i][k] and ques_matrix[i][j] != vocab['<a>'] and ques_matrix[i][j] != 0):
                    similarity_ques = 1.0
                    break
            for l in range(sentence_len):
                if(ans_matrix[i][j] == ques_matrix[i][l] and ans_matrix[i][j] != vocab['<a>'] and ans_matrix[i][j] != 0):
                    similarity_ans = 1.0
                    break
            ques_each_wo.append(similarity_ques)
            ans_each_wo.append(similarity_ans)
        ques_wordoverlap.append(ques_each_wo)
        ans_wordoverlap.append(ans_each_wo)
    return np.asarray(ques_wordoverlap, dtype=theano.config.floatX).reshape((batch_size, 1, sentence_len, 1)),\
           np.asarray(ans_wordoverlap, dtype=theano.config.floatX).reshape((batch_size, 1, sentence_len, 1))

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

def param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, embedding_size + 1)
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True, target = 'dev0'
        )
        tparams['cnn_W_' + str(filter_size)] = W
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True, target = 'dev0')
        tparams['cnn_b_' + str(filter_size)] = b
        grad_params += [W, b]
    return tparams, grad_params

def param_init_softmax(num_filters_total, embedding_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    softmax_shape = (num_filters_total * 2 + 1, 2)
    filter_shape = (num_filters_total, 1, 1, embedding_size)
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W2 = theano.shared(
        np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=softmax_shape),
            dtype=theano.config.floatX
        ),
        borrow=True, target = 'dev3'
    )
    tparams[_p('softmax', 'W')] = W2
    b_values = np.asarray(np.zeros(2), theano.config.floatX)
    b2 = theano.shared(value=b_values, borrow=True, target = 'dev3')
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
    def __init__(self,word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters, learning_rate):
        rng = np.random.RandomState(23455)
        self.params, tparams = [], {}
        # tparams, self.params = param_init_lstm(embedding_size, tparams, self.params)
        tparams, self.params = param_init_cnn(filter_sizes, num_filters, embedding_size, tparams, self.params)
        tparams, self.params = param_init_softmax(len(filter_sizes)*num_filters, embedding_size, tparams, self.params)
        # tparams, self.params = param_init_leaner_transform(embedding_size, tparams, self.params)

        train_q, train_a, y = T.matrix('train_q'), T.matrix('train_a'), T.ivector('y')
        keep_prob = T.scalar('keep_prob')
        wo_q1, wo_a1 = T.tensor4('woq1'), T.tensor4('woa1')

        lookup_table = theano.shared(word_embeddings, borrow=True, target = 'dev1')
        tparams['lookup_table'] = lookup_table
        self.params += [lookup_table]
        self.tparams = tparams

        params = self.params
        n_timesteps = train_q.shape[0]
        n_samples = train_q.shape[1]

        #input1-问题, input2-正向答案, input3-负向答案
        #将每个字替换成字向量
        input_matrix1 = lookup_table[T.cast(train_q.flatten(), dtype="int32")]
        input_matrix2 = lookup_table[T.cast(train_a.flatten(), dtype="int32")]

        #CNN的输入是4维矩阵，这里只是增加了一个维度而已
        input_x1 = input_matrix1.reshape((batch_size, 1, sequence_len, embedding_size))
        input_x2 = input_matrix2.reshape((batch_size, 1, sequence_len, embedding_size))

        #print(input_x1.shape.eval())
        self.dbg_x1 = input_x1

        input_x1 = T.concatenate((input_x1,wo_q1),axis=3)
        input_x2 = T.concatenate((input_x2,wo_a1), axis=3)

        cnn_result_q = self._cnn_net(tparams, input_x1, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)
        cnn_result_a = self._cnn_net(tparams, input_x2, batch_size, sequence_len, num_filters, filter_sizes, embedding_size)

        output_drop1 = self._dropout(rng, cnn_result_q, keep_prob)
        output_drop2 = self._dropout(rng, cnn_result_a, keep_prob)

        W = np.asarray(ortho_weight(num_filters * len(filter_sizes)))
        W_interact = theano.shared(W, borrow=True, target = 'dev2')
        self.tparams[_p('interact', 'W')] = W_interact
        self.params += [W_interact]

        qa_interact, update = theano.scan(lambda a, b, interact: T.dot(T.dot(a, interact), b),
                                     sequences=[output_drop1, output_drop2], non_sequences=W_interact)

        qa_interact = T.reshape(qa_interact, [batch_size, 1])
        qra_interated = T.concatenate((output_drop1, qa_interact, output_drop2), 1)

        probablity = T.nnet.softmax(T.dot(qra_interated, tparams['softmax_W']) + tparams['softmax_b'])

        possibility = probablity[:, 1]
        self.possibility = possibility

        prediction = T.argmax(probablity, 1)
        self.prediction = prediction
        self.cost = -T.mean(T.log(probablity[T.arange(y.shape[0]), y]))
        self.error = T.mean(T.neq(prediction, y))

        grads = T.grad(self.cost, params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [train_q, train_a, y, keep_prob, wo_q1, wo_a1],
            [self.prediction, self.cost, self.error],
            updates=updates
        )

        self.validate_model = theano.function(
            [train_q, train_a, y, keep_prob, wo_q1, wo_a1],
            [self.possibility, self.cost]
        )

    def _dropout(self, rng, layer, keep_prob):
        srng = T.shared_randomstreams.RandomStreams(rng.randint(123456))
        mask = srng.binomial(n=1, p=keep_prob, size=layer.shape)
        output = layer * T.cast(mask, theano.config.floatX)
        output = output / keep_prob
        return output

    def _cnn_net(self, tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, embedding_size):
        outputs = []
        for filter_size in filter_sizes:
            filter_shape = (num_filters, 1, filter_size, embedding_size + 1)
            input_shape = (batch_size, 1, sequence_len, embedding_size + 1)
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
    # cPickle.dump(tparams[_p('lstm', 'W')].get_value(borrow=True), save_file, -1)
    # cPickle.dump(tparams[_p('lstm', 'U')].get_value(borrow=True), save_file, -1)
    # cPickle.dump(tparams[_p('lstm', 'b')].get_value(borrow=True), save_file, -1)

    for filter_size in filter_sizes:
        cPickle.dump(tparams[_p('cnn_W', str(filter_size))].get_value(borrow=True), save_file, -1)
        cPickle.dump(tparams[_p('cnn_b', str(filter_size))].get_value(borrow=True), save_file, -1)

    cPickle.dump(tparams[_p('lookup','table')].get_value(borrow=True), save_file, -1)
    cPickle.dump(tparams[_p('softmax', 'W')].get_value(borrow=True), save_file, -1)
    cPickle.dump(tparams[_p('softmax', 'b')].get_value(borrow=True), save_file, -1)
    cPickle.dump(tparams[_p('interact', 'W')].get_value(borrow=True), save_file, -1)

    # cPickle.dump(tparams[_p('leaner_transform', 'W')].get_value(borrow=True), save_file, -1)

    print "Saved model parameters to %s." % outfile
    save_file.close()

def load_model_parameters(model, path, filter_sizes):
    tparams = model.tparams
    save_file = open(path, 'rb')
    # tparams[_p('lstm', 'W')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('lstm', 'U')].set_value(cPickle.load(save_file), borrow=True)
    # tparams[_p('lstm', 'b')].set_value(cPickle.load(save_file), borrow=True)

    for filter_size in filter_sizes:
        tparams[_p('cnn_W', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)
        tparams[_p('cnn_b', str(filter_size))].set_value(cPickle.load(save_file), borrow=True)

    tparams[_p('lookup','table')].set_value(cPickle.load(save_file), borrow=True)
    tparams[_p('softmax', 'W')].set_value(cPickle.load(save_file), borrow=True)
    tparams[_p('softmax', 'b')].set_value(cPickle.load(save_file), borrow=True)
    tparams[_p('interact', 'W')].set_value(cPickle.load(save_file), borrow=True)

    # tparams[_p('leaner_transform', 'W')].set_value(cPickle.load(save_file), borrow=True)

    print "Building model model from %s " % (path)
    save_file.close()

def train():
    global logfile_path
    global trainfile
    global train0file
    global test1file

    num_filters = 500
    batch_size = 256
    filter_sizes = [1,2,3]
    embedding_size = 300
    sequence_len = 50
    learning_rate = 0.001
    n_epochs = 2000000
    validation_freq = 100
    keep_prob_value = 0.75

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
    log("sequence_len = " + str(sequence_len), logfile_path)
    log("validation_freq = " + str(validation_freq), logfile_path)
    log("keep_prob_value = " + str(keep_prob_value), logfile_path)
    log("train_1_file = " + str(trainfile.split('/')[-1]), logfile_path)
    log("train_0_file = " + str(train0file.split('/')[-1]), logfile_path)
    log("test_file = " + str(test1file.split('/')[-1]), logfile_path)
    log("vector_file = " + str(vectorsfile.split('/')[-1]), logfile_path)

    vocab = build_vocab()
    print len(vocab),'----------'
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    testList = load_test_list(test1file)

    model = QAMODEL(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len, embedding_size=embedding_size,
                  filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=learning_rate)

    epoch = 0
    done_looping = False
    print 'behind epoch=0'
    best_MRR = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_q, train_a, train_mark = load_train_data(trainList, vocab, batch_size, sequence_len)
        wo_q1, wo_a1 = gener_wordoverlap(train_q, train_a, batch_size, sequence_len, vocab)
        prediction_ij, cost_ij, error_ij  = model.train_model(train_q,train_a,train_mark,keep_prob_value, wo_q1, wo_a1)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            MRR_result = validation(model.validate_model, testList, vocab, batch_size, sequence_len)
            if(MRR_result > best_MRR):
                save_model_parameters_theano(model, paramsfile, filter_sizes)
                best_MRR = MRR_result
                print 'best_MRR .............',best_MRR

def predict(modelClass=QAMODEL):
    global logfile_path
    global trainfile
    global train0file
    global test1file

    num_filters = 500
    batch_size = 256
    filter_sizes = [1, 2, 3]
    embedding_size = 300
    sequence_len = 50
    learning_rate = 0.001
    n_epochs = 2000000
    validation_freq = 100

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
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    model = modelClass(word_embeddings=word_embeddings, batch_size=batch_size, sequence_len=sequence_len,
                       embedding_size=embedding_size,
                       filter_sizes=filter_sizes, num_filters=num_filters, learning_rate=learning_rate)

    load_model_parameters(model, paramsfile, filter_sizes)

    print 'start'
    testList = load_test_list(validationfile)
    print 'Evaluation ......'
    validation(model.validate_model, testList, vocab, batch_size, sequence_len)
    print 'end'


if __name__ == '__main__':
    train()
    # predict()
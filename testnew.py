#-*-coding:utf-8-*-
#!/usr/bin/env python

import theano.tensor as T
import theano
import numpy as np

# def gener_wordoverlap(ques_matrix, ans_matrix, batch_size, sentence_len):
#     ques_wordoverlap = []
#     ans_wordoverlap = []
#     for i in range(batch_size):
#         ques_each_wo = []
#         ans_each_wo = []
#         for j in range(sentence_len):
#             similarity_ques = 0.0
#             similarity_ans = 0.0
#             for k in range(sentence_len):
#                 if(ques_matrix[i][j] == ans_matrix[i][k] and ques_matrix[i][j] != 1):
#                     similarity_ques = 1.0
#                     break
#             for l in range(sentence_len):
#                 if(ans_matrix[i][j] == ques_matrix[i][l]):
#                     similarity_ans = 1.0
#                     break
#             ques_each_wo.append(similarity_ques)
#             ans_each_wo.append(similarity_ans)
#         ques_wordoverlap.append(ques_each_wo)
#         ans_wordoverlap.append(ans_each_wo)
#     return np.asarray(ques_wordoverlap),np.asarray(ans_wordoverlap)

# a = T.tensor4('a')
# b = T.tensor4('b')
#
# result = T.concatenate((a,b),3)
# f = theano.function([a,b],result)
#
# aa = np.asarray([[[[1,1]]],[[[1,1]]]])
# bb = np.asarray([[[[2]]],[[[2]]]])
#
#
# rr = f(aa,bb)
#
# print rr


# coefficients=T.vector('coeff')
# x = T.iscalar('x')
# sum_poly_init = T.scalar('sum_poly')
# result, update = theano.scan(lambda coefficients, power, sum_poly, x: T.cast(sum_poly +
#                              coefficients*(x**power),dtype=theano.config.floatX),
#                              sequences=[coefficients, T.arange(coefficients.size)],
#                             outputs_info=[sum_poly_init],
#                             non_sequences=[x])
#
# poly_fn = theano.function([coefficients,sum_poly_init,x], result, updates=update)
#
# coeff_value = np.asarray([1,3,6,5], dtype=theano.config.floatX)
# x_value = 3
# poly_init_value = 0.
# print poly_fn(coeff_value,poly_init_value, x_value)


# A = T.matrix('A')
# B = T.matrix('B')
# interact = T.matrix('interact')
#
# result, update = theano.scan(lambda a, b, interact: T.dot(T.dot(a, interact), b),
#                              sequences=[A, B], non_sequences=interact)
# output = result
# ff = theano.function([A, B, interact], output, updates=update)
#
# a = np.asarray([[1,1],[1,1]])
# b = np.asarray([[2,2],[2,2]])
# i = np.asarray([[1,0],[2,0]])
#
# rr = ff(a,b,i)
# print [rr]



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

a = np.asarray([[1,1],[1,1],[1,1]])
b = np.asarray([[2,2],[2,2],[2,2]])

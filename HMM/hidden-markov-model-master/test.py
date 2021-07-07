# -*- coding:utf-8 -*-
# Filename: test_weather.py
# Author：hankcs
# Date: 2016-08-06 PM6:04
import numpy as np
import hmm

def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label

def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list

def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v

def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m


##
def simulate(self, T):
    '''
    input: 马尔科夫参数 A,B,pi ; 观测序列长度
    output: 生成观测序列
    '''
    def draw_from(probs):
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]
 
    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(self.pi)
    observations[0] = draw_from(self.B[states[0],:])
    for t in range(1, T):
        states[t] = draw_from(self.A[states[t-1],:])
        observations[t] = draw_from(self.B[states[t],:])
    return observations,states

if __name__=='__main__':
    states = ('Healthy', 'Fever')
    observations = ('normal', 'cold', 'dizzy')
    start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    transition_probability = {
        'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
        'Fever': {'Healthy': 0.4, 'Fever': 0.6},
    }
    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }
    states_label_index, states_index_label = generate_index_map(states)
    observations_label_index, observations_index_label = generate_index_map(observations)
    A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
    print (A)
    B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
    print(B)
    observations_index = convert_observations_to_index(observations, observations_label_index)
    pi = convert_map_to_vector(start_probability, states_label_index)
    print (pi)
    ###生产隐马尔可夫模型
    h = hmm.HMM(A, B, pi)
    ## 随机生成观测序列，状态序列
    observations_data, states_data = h.simulate(10)
    print (observations_data)
    print (states_data)
    ## 前向算法
    forward_matrix=h._forward(observations_data)
    forward_prob=h.observation_prob(observations_data)
    ## 逆向算法
    backward_matrix=h._backward(observations_data)
    backward_prob=np.sum(backward_matrix[:,1])
    ## 维特比算法
    observations_data, states_data = h.simulate(10)     
    V, p = h.viterbi(observations_data)
    print (" " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_data))
    for s in range(0, 2):
        print ( "%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s]))
    print ( '\nThe most possible states and probability are:')
    p, ss = h.state_path(observations_data)
    for s in ss:
        print  (states_index_label[s],)
    print (p)
    
    ## Baum-Welch参数估计  
    # run a baum_welch_train
    observations_data, states_data = h.simulate(100)
    # print observations_data
    # print states_data
    guess = hmm.HMM(np.array([[0.5, 0.5],
                              [0.5, 0.5]]),
                    np.array([[0.3, 0.3, 0.3],
                              [0.3, 0.3, 0.3]]),
                    np.array([0.5, 0.5])
                    )
    guess.baum_welch_train(observations_data)
    states_out = guess.state_path(observations_data)[1]
    p = 0.0
    for s in states_data:
        if next(states_out) == s: p += 1

    print (p / len(states_data))

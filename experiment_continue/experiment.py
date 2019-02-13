import sys
import os
sys.path.append('/home/kirilman/Projects/nir/nir/')
sys.path.append('/home/kirill/Projects/NIR')
sys.path.append('/home/kirill/Projects/NIR')

import sequence_generator as generator
import numpy as np
import matplotlib.pylab as plt
from pomegranate import HiddenMarkovModel, DiscreteDistribution, MarkovChain, NormalDistribution
from myutils import frequency_occurrence
plt.rcParams.update({'font.size': 17})
import gc
from myutils import print_model_distribution
import myutils
from multiprocessing.dummy import Pool as ThreadPool


def experiment(model, normal_seq, anormal_seq, alpha, params, mean, variance, N=150, num_launch = 0):

    normal_score = []
    anormal_score = []
    normal_score+=[model.log_probability(normal_seq)]
#     print(model.distributions[1])
    anormal_score+=[model.log_probability(anormal_seq)]
    print(anormal_score[0])
    if anormal_score[0] == float('-inf'):   
        anormal_score[0] = -1500
    # print("Нормальный {}, Аномальный {}".format(normal_score[0],anormal_score[0]))

    #Построение графика
    # fig_seq = plt.figure(dpi = 150)
    # plt.plot(alpha,'w')
    # plt.plot(anormal_seq,'r')    
    # plt.plot(normal_seq,'b')
    # plt.plot(normal_seq,'b.')
    # plt.grid()
    # plt.savefig('Непрерывный'+str(num_launch)+'.png',dpi = 150)
    # # plt.show()
    
    
    log_prob_arr = []
    for i in range(100):
        sequence = generator.Sequence( N, alpha, type = 'continue', params=params, mean = mean, variance = variance)
        seq = sequence.sequence
        if i%50 == 0:
            fig_sequence = plt.figure(dpi = 140)
            plt.plot(alpha,'w')
            plt.plot(seq,'b.')
            plt.plot(seq,'b')
            plt.savefig('/home/kirilman/Projects/nir/nir/experiment_continue/Graphs/sequence_graph__'+str(i)+'.png')
        log_prob_arr += [ model.log_probability(seq)]

    plt.close()

    fig_log = plt.figure(figsize=(3.5,10))
    plt.plot([1]*len(log_prob_arr), log_prob_arr, '.',markersize = 15)
    plt.plot([1], anormal_score, 'r.',markersize = 15)
    # plt.plot([1], normal_score, 'g.',markersize = 15)
    plt.ylabel('log probability')
    plt.xlim(0.95, 1.05)
    # plt.tight_layout()
    # plt.savefig('График log_probability'+str(num_launch)+'.png', dpi = 140)


# Совместный график
    fig_sub = plt.figure(figsize = (15,4))

    ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    ax2.plot([1] * len(log_prob_arr), log_prob_arr, '.', markersize=15)
    ax2.plot([1], anormal_score, 'r*', markersize=15)
    # ax2.plot([1], normal_score, 'g*', markersize=12)

    ax2.set_ylabel('log probability')
    # ax2.set_xlim(0.9, 1.2)
    ax2.set_xticks([0.95,1,1.05])
    ax2.set_xticklabels(['','1',''])

    ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    # ax.plot(alpha,'w')
    ax.plot(anormal_seq,'r.')
    ax.plot(anormal_seq,'r')
    ax.plot(normal_seq,'b')
    ax.plot(normal_seq,'b.')
    ax.grid()
    # ax.set_yticks(range(len(alpha)))

# plt.tight_layout()
    plt.savefig('/home/kirilman/Projects/nir/nir/experiment_continue/Graphs/Log_and_seq/gh_'+str(num_launch)+'.png',dpi=100)
    plt.close('all')
    return model

def write_log(N, N_tr, norm_par, norm_mean, norm_var, an_params, anomal_mean, anomal_var):
    with open('log_experiment.txt','w') as file:
        file.write('Параметры :\n')
        file.write('    Длина сигнала = {}, длина сиг.обучения = {}\n'.format(N, N_tr))
        file.write('Параметры нормального сигнала:\n')
        file.write(str(norm_par)+'\n')
        file.write('    means = {}, var = {} \n'.format(norm_mean, norm_var))
        file.write('Параметры аномального сигнала:\n')
        file.write('    '+str(an_params)+'\n')
        file.write('    means = {}, var = {} \n'.format(anomal_mean, anomal_var))
        file.close()
if __name__ ==  "__main__":
    N = 3000 # длина сигнала
    N_train = 3000
    n_comp = 3
    count_launch = 15
    # norm_params = {'a': {'len': [10, 20], 'depend_on': False},
    #                'b': {'len': [30, 50], 'depend_on': False},
    #                'c': {'len': [50, 200],'depend_on': False}}

    # an_params =   {'a': {'len': [20, 40], 'depend_on': False},
    #                'b': {'len': [30, 120], 'depend_on': False},
    #                'c': {'len': [40, 50], 'depend_on': False}}
    norm_params = {'a': {'len': [10, 20], 'depend_on': False},
                   'b': {'len': [30, 50], 'depend_on': False},
                   'c': {'len': [50, 80],'depend_on': False}}

    an_params =   {'a': {'len': [10, 20], 'depend_on': False},
                   'b': {'len': [30, 150], 'depend_on': False},
                   'c': {'len': [50, 60], 'depend_on': False}}

    alpha = ['a','b','c']
    mean = [0, 0.5, 0.8] ; variance = [0.3, 0.2, 0.1]
    anomal_mean = mean; anomal_variance = variance
    write_log(N,N_train, norm_params, mean, variance, an_params, anomal_mean, anomal_variance)

    for k in range(count_launch):
        np.random.seed(k) 
        sequence = generator.Sequence( N_train, alpha, type = 'continue',
                                            params = norm_params,
                                            mean = mean , variance = variance)
        normal_signal = sequence.sequence
        labels = list(map(myutils.rename_state,sequence.path))
        an_sequence = generator.Sequence( N, alpha, type = 'continue',
                                            params=an_params,
                                            mean=anomal_mean, variance=anomal_variance)
        
        anormal_signal = an_sequence.sequence
        
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [normal_signal],
                                            labels = [labels], algorithm='labeled')

        print('Параметры {}, {}, {}'.format(N, mean, variance))
        experiment(model, normal_signal, anormal_signal, ['a','b','c'], norm_params, mean,variance, N , k)

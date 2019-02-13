import sys
import os
sys.path.append('/home/kirilman/Projects/nir/nir/')
sys.path.append('/home/kirill/Projects/nir')


import sequence_generator as generator
import numpy as np
# import matplotlib.pylab as plt
from pomegranate import HiddenMarkovModel, DiscreteDistribution, MarkovChain, NormalDistribution
from myutils import frequency_occurrence
import gc
from myutils import print_model_distribution
import myutils
# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

def experiment(model, normal_seq, anomal_seq, alpha, norm_params, mean, variance, 
                anomal_params, anomal_mean, anomal_var, N=150, num_launch = 0):   
    
    normal_score = []
    anomal_score = []
    normal_score+=[model.log_probability(normal_seq)]
#     print(model.distributions[1])
    anomal_score+=[model.log_probability(anomal_seq)]
    print(anomal_score[0])
    if anomal_score[0] == float('-inf'):   
        anomal_score[0] = -1500
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
        sequence = generator.Sequence( N, alpha, type = 'continue', params=norm_params, mean = mean, variance = variance)
        seq = sequence.sequence
        log_prob_arr += [ model.log_probability(seq)]

        sequence = generator.Sequence( N, alpha, type = 'continue', params=anomal_params, mean = anomal_mean,
                                    variance = anomal_var)
        anomal_seq = sequence.sequence
        anomal_score += [ model.log_probability(anomal_seq)]
        # if i%50 == 0:
        #     fig_sequence = plt.figure(dpi = 140)
        #     # plt.plot(alpha,'w')
            # plt.plot(seq,'b.')
            # plt.plot(seq,'b')
            # plt.savefig('/home/kirilman/Projects/nir/nir/experiment_continue/Graphs/sequence_graph_l_'+str(num_launch)+str(i)+'.png')
    # plt.close()

    fig_log = plt.figure(num = num_launch, figsize=(3.5,10))

    plt.plot([1]*len(log_prob_arr), log_prob_arr, '.',markersize = 12)
    plt.plot([1]*len(anomal_score), anomal_score, 'r.',markersize = 12)
    # plt.plot([1], normal_score, 'g.',markersize = 15)
    plt.ylabel('log probability')
    plt.xlim(0.95, 1.05)
    # plt.tight_layout()
    # plt.savefig('График log_probability'+str(num_launch)+'.png', dpi = 140)


# Совместный график
    fig_sub = plt.figure(figsize = (16,5))

    ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    ax2.plot([1] * len(log_prob_arr), log_prob_arr, '.', markersize=10)
    ax2.plot([1]*len(anomal_score), anomal_score, 'rx', markersize=10)
    # ax2.plot([1], normal_score, 'g*', markersize=12)

    ax2.set_ylabel('log probability')
    # ax2.set_xlim(0.9, 1.2)
    ax2.set_xticks([0.95,1,1.05])
    ax2.set_xticklabels(['','1',''])

    ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    ax.plot(anomal_seq,'r')
    ax.plot(normal_seq,'b')
    ax.grid()
    # ax.set_yticks(range(len(alpha)))

    # plt.tight_layout()
    plt.savefig('Graphs/Log_and_seq/gh_'+str(num_launch)+'.png',dpi=150)
    plt.close()
    return model

def f(model, normal_signal,N_train, N, alpha, norm_params, mean, variance, an_params, anomal_mean, anomal_variance,
       n_comp,launch):

    np.random.seed(launch) 


    an_sequence = generator.Sequence( N, alpha, type = 'continue',
                                        params=an_params,
                                        mean=anomal_mean, variance=anomal_variance)
    
    anormal_signal = an_sequence.sequence
    
    
    print(model.states)
    print('Параметры {}, {}, {}'.format(N, mean, variance))
    t = experiment(model, normal_signal, anormal_signal, ['a','b','c'], norm_params, mean,variance, 
                   an_params, anomal_mean, anomal_variance, N, launch)

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
    N = 2000 # длина сигнала
    N_pool = 10
    N_train = 2000
    n_comp = 3
    count_launch = 10
    # norm_params = {'a': {'len': [10, 20], 'depend_on': False},
    #                'b': {'len': [30, 50], 'depend_on': False},
    #                'c': {'len': [50, 200],'depend_on': False}}

    # an_params =   {'a': {'len': [20, 40], 'depend_on': False},
    #                'b': {'len': [30, 120], 'depend_on': False},
    #                'c': {'len': [40, 50], 'depend_on': False}}
    norm_params = {'a': {'len': [20, 21], 'depend_on': False},
                   'b': {'len': [30, 31], 'depend_on': False},
                   'c': {'len': [30, 31],'depend_on': False}}

    an_params =   {'a': {'len': [20, 21], 'depend_on': False},
                   'b': {'len': [100, 101], 'depend_on': False},
                   'c': {'len': [30, 31], 'depend_on': False}}

    alpha = ['a','b','c']

    mean = [0, 0.3, 0.6] ; variance = [0.01, 0.01, 0.01] + np.array([0.03]*3)
    anomal_mean = mean  
    # anomal_mean = mean + np.array([0,0.5,0]) 
    anomal_variance = variance 

    write_log(N, N_train, norm_params, mean, variance, an_params, anomal_mean, anomal_variance)

    sequence = generator.Sequence( N_train, alpha, type = 'continue',
                                        params = norm_params,
                                        mean = mean , variance = variance)
    normal_signal = sequence.sequence
    labels = list(map(myutils.rename_state,sequence.path))
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [normal_signal],
                                        labels = [labels], algorithm='labeled')
    fig = plt.figure(num = 1000, figsize=(15,4))
    plt.plot(normal_signal,'b')
    plt.plot([x / 3 for x in sequence.path], 'r')
    plt.savefig('Graphs/path.png')
    plt.close('all')
    pool = Pool(N_pool)
    with open('model.txt', 'w') as file:
        out = print_model_distribution(model)
        file.write(out)
    params = [(model, normal_signal, N_train, N, alpha, norm_params, mean, variance, an_params, anomal_mean, anomal_variance, n_comp, i+1)
              for i in range(count_launch)]


    start_time = time.time()
    res = pool.starmap(f,params)

    pool.close()
    pool.join()
    print('Time ',time.time() - start_time)

    # for k in range(count_launch):
    #     np.random.seed(k) 
    #     sequence = generator.Sequence( N_train, alpha, type = 'continue',
    #                                         params = norm_params,
    #                                         mean = mean , variance = variance)
    #     normal_signal = sequence.sequence
    #     labels = list(map(myutils.rename_state,sequence.path))
    #     an_sequence = generator.Sequence( N, alpha, type = 'continue',
    #                                         params=an_params,
    #                                         mean=anomal_mean, variance=anomal_variance)
        
    #     anormal_signal = an_sequence.sequence
        
    #     model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [normal_signal],
    #                                         labels = [labels], algorithm='labeled')

    #     print('Параметры {}, {}, {}'.format(N, mean, variance))
    #     experiment(model, normal_signal, anormal_signal, ['a','b','c'], norm_params, mean,variance, N , k)


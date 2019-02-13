import sys
import os
sys.path.append('/home/kirill/Projects/NIR')
sys.path.append('/home/kirill/Projects/NIR')
sys.path.append('/home/kirilman/Projects/nir/nir/')

import sequence_generator as generator
import numpy as np
import matplotlib.pylab as plt
from pomegranate import HiddenMarkovModel, DiscreteDistribution, MarkovChain
from myutils import frequency_occurrence
plt.rcParams.update({'font.size': 17})
import gc
from myutils import print_model_distribution
import myutils

def get_slice(s):
    """
    Получить случайно выбранную подпоследовательность из одного и того же символа с
    Returns:
      start : индекс начала подпоследовательности
      stop : индекс окончания
      с : символ       
    """
    m = np.random.choice(range(len(s)))
    # m = len(s) - 1

    c = s[m]
    start, stop = 0,0
    i = 1 
    flag_1 = True
    flag_2 = True
    while((flag_1==True) or (flag_2 == True)):
        if flag_1 == True:
            if m-i-1 == 0:
                start = m-i+1
                flag_1 = False
            else:
                if s[m - i]!=c:
                    start = m-i+1
                    flag_1 = False
        if flag_2 == True:
            if m + i - 1 == len(s) - 1:
                stop = m + i
                flag_2 = False
            else:
                if s[m+i]!=c:
                    stop = m + i
                    flag_2 = False
        i+=1
    print(start, stop, len(s))
    return start, stop, c   

def experiment_discret(model, normal_seq, anormal_seq, N=150,alpha = ['a','b','c','d','e'],p = 0.05, num_launch = 0):
#     params = {'a': {'len': [1, 1], 'depend_on': False},
#               'b': {'len': [1, 1], 'depend_on': False},
#               'c': {'len': [0, 1], 'depend_on': False},
#               'd': {'len': [0, 1], 'depend_on': 'c' },
#               'e': {'len': [1, 3], 'depend_on': 'b'}}
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
    fig_seq = plt.figure(dpi = 150)
    plt.plot(alpha,'w')
    plt.plot(anormal_seq,'r')    
    plt.plot(normal_seq,'b')
    plt.plot(normal_seq,'b.')
    plt.grid()
    plt.savefig('Дискретный с невозможным переходом'+str(num_launch)+'.png',dpi = 150)
    # plt.show()

    
    log_prob_arr = []
    for i in range(100):
        sequence = generator.Sequence(N,alpha,type='test_discret',p=[0.05,0.1,0.4,0.8])
        seq = sequence.sequence
        if i%50 == 0:
            fig_sequence = plt.figure(dpi = 140)
            plt.plot(alpha,'w')
            plt.plot(seq,'b.')
            plt.plot(seq,'b')
            plt.savefig('/home/kirilman/Projects/nir/nir/experiment_discret/Graphs/sequence_graph__'+str(i)+'.png')
        log_prob_arr += [ model.log_probability(seq)]
    plt.close()

    fig_log = plt.figure(figsize=(3.5,10))
    plt.plot([1]*len(log_prob_arr), log_prob_arr, '.',markersize = 15)
    plt.plot([1], anormal_score, 'r.',markersize = 15)
    # plt.plot([1], normal_score, 'g.',markersize = 15)
    plt.ylabel('log probability')
    plt.xlim(0.95, 1.05)
    # plt.tight_layout()
    plt.savefig('График log_probability'+str(num_launch)+'.png', dpi = 140)
    # plt.show()
    #print(model.to_json())
    
#     with open('experiment1.txt','w') as file:
#         table = myutils.table_from_MarkovChain(model)
#         for t in table:
#             file.write(str(t)+'\n')
#     print(model.distributions[1])


# Совместный график
    fig_sub = plt.figure(figsize = (10,6))

    ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    ax2.plot([1] * len(log_prob_arr), log_prob_arr, '.', markersize=15)
    ax2.plot([1], anormal_score, 'r*', markersize=15)
    ax2.plot([1], normal_score, 'g*', markersize=12)

    ax2.set_ylabel('log probability')
    # ax2.set_xlim(0.9, 1.2)
    ax2.set_xticks([0.95,1,1.05])
    ax2.set_xticklabels(['','1',''])

    ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    ax.plot(alpha,'w')
    ax.plot(anormal_seq,'r.')
    ax.plot(anormal_seq,'r')
    ax.plot(normal_seq,'b')
    ax.plot(normal_seq,'b.')
    ax.grid()
    ax.set_yticks(range(len(alpha)))
    ax2.set_xticklabels(alpha)

# plt.tight_layout()
    plt.savefig('/home/kirilman/Projects/nir/nir/experiment_discret/Graphs/Log_and_seq/gh_'+str(num_launch)+'.png',dpi=100)
    plt.close('all')
    return model

if __name__ == "__main__":
    N = 150
    arr_seqs = []
    N_train = 3000
    count = 25
    file = open('result.txt','w')
    for i in range(count):
        np.random.seed(i)  # Для случайной инициализации модели
        alpha = ['a','b','c','d','e']
        sequence = generator.Sequence(N_train,alpha,type='test_discret',p=[0.05,0.1,0.4,0.8])

        
        normal_seq = sequence.sequence.copy()
        print(normal_seq[:15])

        arr_seqs += [normal_seq]
    #Аномальная последовательность
    #     anormal_seq = sequence.anormal(p)
        anormal_seq = normal_seq[:N].copy()

        start, stop, simbol = get_slice(normal_seq[:N])
        if stop == len(anormal_seq):
            # anormal_seq[start:stop] = [normal_seq[start-1]]*(stop - start)
            new_s = normal_seq[np.random.choice(range(100))]
            while new_s == simbol:
                new_s = normal_seq[np.random.choice(range(100))]
            anormal_seq[start:stop] = [new_s]*(stop - start)

        else:
            # anormal_seq[start:stop] = [normal_seq[stop+1]]*(stop - start)
            new_s = normal_seq[np.random.choice(range(100))]
            while new_s == simbol:
                new_s = normal_seq[np.random.choice(range(100))]
            anormal_seq[start:stop] = [new_s]*(stop - start)

        start, stop, simbol = get_slice(normal_seq[:N])
        if stop == len(anormal_seq):
            # anormal_seq[start:stop] = [normal_seq[start-1]]*(stop - start)
            new_s = normal_seq[np.random.choice(range(100))]
            while new_s == simbol:
                new_s = normal_seq[np.random.choice(range(100))]
            anormal_seq[start:stop] = [new_s]*(stop - start)

        else:
            # anormal_seq[start:stop] = [normal_seq[stop+1]]*(stop - start)
            new_s = normal_seq[np.random.choice(range(100))]
            while new_s == simbol:
                new_s = normal_seq[np.random.choice(range(100))]
            anormal_seq[start:stop] = [new_s]*(stop - start)
    #     n_count = 5
    #     anormal_seq[20:20+n_count] = ['b']*n_count
        
        # print('Длина нормальной ',len(normal_seq),', аномальной ', len(anormal_seq))
    #Модель

        # model_hmm = MarkovChain.from_samples([normal_seq]);
        gc.collect()
        print(normal_seq[-5:])
        labels = list(map(myutils.rename_state,sequence.path))
        # plt.plot(normal_seq)
        # # break
        model_hmm = HiddenMarkovModel.from_samples(DiscreteDistribution,n_components = len(alpha),X=[normal_seq],
                                                    labels = [labels] , algorithm = 'labeled');
        # model_hmm = HiddenMarkovModel.from_samples(DiscreteDistribution,n_components = len(alpha),X=[normal_seq]);

        # model_hmm.bake()
        experiment_discret(model = model_hmm,normal_seq =  normal_seq[:N],anormal_seq = anormal_seq,N = N, num_launch=i)

# Вывод в файл
        file.write(str(i)+'\n')
        
        if isinstance(model_hmm,HiddenMarkovModel):
            out = print_model_distribution(model_hmm)
        else:
            out = str(myutils.table_from_MarkovChain(model_hmm))
        file.write(out)
        print(id(model_hmm))      
        print('seq',id(sequence))
        del(model_hmm)
        del sequence
        # print(model_hmm)
    
        gc.collect()
    file.close()
    for s in arr_seqs:
        plt.plot(s)
    plt.show()
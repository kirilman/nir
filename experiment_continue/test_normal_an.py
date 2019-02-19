import sys
import os
sys.path.append('/home/kirilman/Projects/nir/nir/')
sys.path.append('/home/kirill/Projects/nir')


import sequence_generator as generator
import numpy as np
# import matplotlib.pylab as plt
from pomegranate import HiddenMarkovModel, DiscreteDistribution, MarkovChain, NormalDistribution
from myutils import print_model_distribution
import myutils
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import json

def parsing_json(fname):
    with open(fname,'r') as file:
        # json_string = file.read()
        pars_string = json.load(file)
    return pars_string['array']

def run(arg,k):
    np.random.seed(k)
    exp_type = arg['type']
    N = arg['N']; alpha = arg['alpha']; n_comp = arg['n_comp']
    norm_params = arg['norm_params']
    an_params = arg['an_params']
    save_dir = arg['dir']
    mean = arg['mean']; variance = arg['varience']
    anomal_mean = arg['anomal_mean'];
    anomal_variance = arg['anomal_varience']
    norm_gen = generator.Sequence( N, alpha, type = exp_type,
                                        params = norm_params,
                                        mean = mean , variance = variance)
    norm_signal = norm_gen.sequence

    an_gen = generator.Sequence( N, alpha, type = exp_type,
                                        params = an_params,
                                        mean = anomal_mean, variance = anomal_variance)
                        
    an_signal = an_gen.sequence

    # an_signal[180:200] = np.random.normal(2,0.02,20)
    an_labels = list(map(myutils.rename_state,an_gen.path))
    labels = list(map(myutils.rename_state,norm_gen.path))
    if exp_type == 'continue':    
        model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [norm_signal],
                                            labels = [labels], algorithm='labeled')

        an_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [an_signal],
                                        labels = [an_labels], algorithm='labeled')
    else:
        model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = n_comp, X = [norm_signal],
                                            labels = [labels], algorithm='labeled')

        an_model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = n_comp, X = [an_signal])
    #     model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = n_comp, X = [norm_signal])
    #     an_model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = n_comp, X = [an_signal])
        
    l1 = model.log_probability(norm_signal)
    l2 = model.log_probability(an_signal)
    cdir = os.getcwd()
    path = cdir + '/'+arg['dir']
    try:
        os.mkdir(path)
    except:
        pass
    with open(path+'/log_'+str(k)+'.txt','w') as file:
        out = myutils.print_model_distribution(model)
        file.write(out)
        out = myutils.print_model_distribution(an_model)
        file.write(out)
        file.write('l_normal = {} l_anomal = {}'.format(l1, l2))
        # out = myutils.print_model_distribution(model_2)
        # file.write(out)
        # file.write(str(an_model.to_json()))


    # fig_sub = plt.figure(figsize = (18,5.9))
    fig_sub = plt.figure(figsize = (16,5.9))

    ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    ax2.plot([1] * len([l1]), l1, 'b.', markersize=12)
    ax2.plot([1]*len([l2]), l2, 'r.', markersize=12)
    # ax2.plot([1], normal_score, 'g*', markersize=12)

    ax2.set_ylabel('log probability')
    # ax2.set_xlim(0.9, 1.2)
    ax2.set_xticks([0.95,1,1.05])
    ax2.set_xticklabels(['','',''])

    ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    ax.plot(norm_signal,'b',label='Normal')  #Ошибка в цветах
    ax.plot(an_signal,'r',label='Abnormal')
    ax.set_xlabel('Time',)
    # ax.grid()
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path+'/plot'+str(k)+'.png',dpi=180)

    plt.close()
    # ax.set_y
    print(' {}, {}'.format(l1, l2))
    print('На аномальной')

    l1 = an_model.log_probability(norm_signal)
    l2 = an_model.log_probability(an_signal)
    print(' {}, {}'.format(l1, l2))
    print(' Норма\n {}'.format(model.predict_proba(an_signal)))


if __name__ == "__main__":
    args = parsing_json('/home/kirilman/Projects/nir/nir/experiment_continue/config.json')
    for k, arg in enumerate(args):
        run(arg,k)
    # N = 1000
    # parsing_json('/home/kirilman/Projects/nir/nir/experiment_continue/config.json')
    # # alpha = ['a','b','c']
    # alpha = ['a','b']
    # n_comp = 2
    # norm_params = {'a': {'len': [25, 25], 'depend_on': False},
    #                'b': {'len': [30, 30], 'depend_on': False}}

    # an_params =   {'a': {'len': [25, 25], 'depend_on': False},
    #                'b': {'len': [80, 80], 'depend_on': False}}
   


    # mean = [0, 1, 2 ]; variance = [0.01, 0.2, 0.2]
    # mean = [0, 2]; variance = [0.1, 0.1] 
    # anomal_mean = mean 
    # # anomal_mean = mean + np.array([0,0.2,0]) 
    # anomal_variance = variance 
    # # anomal_variance = variance + np.array([0.02*3]) 
    # norm_gen = generator.Sequence( N, alpha, type = 'continue',
    #                                     params = norm_params,
    #                                     mean = mean , variance = variance)
    # norm_signal = norm_gen.sequence

    # an_gen = generator.Sequence( N, alpha, type = 'continue',
    #                                     params = an_params,
    #                                     mean = anomal_mean, variance = anomal_variance)
                        
    # an_signal = an_gen.sequence

    # # an_signal[180:200] = np.random.normal(2,0.02,20)
    # an_labels = list(map(myutils.rename_state,an_gen.path))
    # labels = list(map(myutils.rename_state,norm_gen.path))
    # model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [norm_signal],
    #                                     labels = [labels], algorithm='labeled')

    # model_2 = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [norm_signal],
    #                                     labels = [labels], algorithm='labeled')

    # an_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [an_signal],
    #                                     labels = [an_labels], algorithm='labeled')

    # l1 = model.log_probability(norm_signal)
    # l2 = model.log_probability(an_signal)
    # with open('/home/kirilman/Projects/nir/nir/experiment_continue/test_exp_res/log.txt','w') as file:
    #     out = myutils.print_model_distribution(model)
    #     file.write(out)
    #     out = myutils.print_model_distribution(an_model)
    #     file.write(out)
    #     file.write('l_normal = {} l_anomal = {}'.format(l1, l2))
    #     # out = myutils.print_model_distribution(model_2)
    #     # file.write(out)
    #     # file.write(str(an_model.to_json()))


    # # fig_sub = plt.figure(figsize = (18,5.9))
    # fig_sub = plt.figure(figsize = (16,5.9))

    # ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    # ax2.plot([1] * len([l1]), l1, 'b.', markersize=12)
    # ax2.plot([1]*len([l2]), l2, 'r.', markersize=12)
    # # ax2.plot([1], normal_score, 'g*', markersize=12)

    # ax2.set_ylabel('log probability')
    # # ax2.set_xlim(0.9, 1.2)
    # ax2.set_xticks([0.95,1,1.05])
    # ax2.set_xticklabels(['','',''])

    # ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    # ax.plot(norm_signal,'b',label='Normal')  #Ошибка в цветах
    # ax.plot(an_signal,'r',label='Abnormal')
    # ax.set_xlabel('Time',)
    # # ax.grid()
    # plt.legend(loc=1)
    # plt.tight_layout()
    # plt.savefig('Graphs/plot.png',dpi=180)
    # plt.show()
    # plt.close()
    # # ax.set_y
    # print(' {}, {}'.format(l1, l2))
    # print('На аномальной')

    # l1 = an_model.log_probability(norm_signal)
    # l2 = an_model.log_probability(an_signal)
    # print(' {}, {}'.format(l1, l2))
    # print(' Норма\n {}'.format(model.predict_proba(an_signal)))

    # N = 1000
    # alpha = ['a','b','c','d']
    # n_comp = 4
    # norm_params = {'a': {'len': [15, 15], 'depend_on': False},
    #                'b': {'len': [20, 20], 'depend_on': False},
    #                'c': {'len': [20, 20],'depend_on': False},
    #                'd': {'len': [20, 20],'depend_on': False}}

    # an_params =   {'a': {'len': [15, 15], 'depend_on': False},
    #                'b': {'len': [210, 210], 'depend_on': False},
    #                'c': {'len': [60, 60], 'depend_on': False},
    #                'd': {'len': [20, 20],'depend_on': False}}

    # mean = [0, 1, 2 , 1.5 ]; variance = [0.01, 0.01, 0.01, 0.01] + np.array([0.01]*4)
    # anomal_mean = mean 
    # # anomal_mean = mean + np.array([0,0.2,0]) 
    # anomal_variance = variance

    # norm_gen = generator.Sequence( N, alpha, type = 'continue',
    #                                     params = norm_params,
    #                                     mean = mean , variance = variance)
    # norm_signal = norm_gen.sequence

    # an_gen = generator.Sequence( N, alpha, type = 'continue',
    #                                     params = an_params,
    #                                     mean = anomal_mean, variance = anomal_variance)
    # an_signal = an_gen.sequence
    # an_labels = list(map(myutils.rename_state,an_gen.path))
    # labels = list(map(myutils.rename_state,norm_gen.path))
    # model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [norm_signal],
    #                                     labels = [labels], algorithm='labeled')

    # model_2 = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [norm_signal],
    #                                     labels = [labels], algorithm='labeled')

    # an_model = HiddenMarkovModel.from_samples(NormalDistribution, n_components = n_comp, X = [an_signal],
    #                                     labels = [an_labels], algorithm='labeled')
    # with open('/home/kirilman/Projects/nir/nir/experiment_continue/test_exp_res/log.txt','w') as file:
    #     out = myutils.print_model_distribution(model)
    #     file.write(out)
    #     out = myutils.print_model_distribution(an_model)
    #     file.write(out)
    #     out = myutils.print_model_distribution(model_2)
    #     file.write(out)
    #     file.write(str(an_model.to_json()))
    # l1 = model.log_probability(norm_signal)
    # l2 = model.log_probability(an_signal)

    # fig_sub = plt.figure(figsize = (16,5))

    # ax2 = fig_sub.add_axes([0.12, 0.1, 0.07, 0.8])
    # ax2.plot([1] * len([l1]), l1, '.', markersize=10)
    # ax2.plot([1]*len([l2]), l2, 'rx', markersize=10)
    # # ax2.plot([1], normal_score, 'g*', markersize=12)

    # ax2.set_ylabel('log probability')
    # # ax2.set_xlim(0.9, 1.2)
    # ax2.set_xticks([0.95,1,1.05])
    # ax2.set_xticklabels(['','',''])

    # ax = fig_sub.add_axes([0.24, 0.1, 0.74, 0.8])
    # ax.plot(norm_signal,'b',label='Норм.')  #Ошибка в цветах
    # ax.plot(an_signal,'r',label='Аномал.')
    # ax.grid()
    # plt.legend()
    # plt.savefig('Graphs/plot.png',dpi=150)
    # plt.close()
    # # ax.set_y
    # print(' {}, {}'.format(l1, l2))
    # print('На аномальной')

    # l1 = an_model.log_probability(norm_signal)
    # l2 = an_model.log_probability(an_signal)
    # print(' {}, {}'.format(l1, l2))

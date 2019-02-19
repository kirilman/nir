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

def run_test(arg, k):
    np.random.seed(k)
    exp_type = arg['type']
    N = arg['N']; alpha = arg['alpha']; n_comp = arg['n_comp']
    norm_params = arg['norm_params']
    save_dir = arg['dir']
    
    sequence = generator.Sequence( N, alpha, type = exp_type,
                                        params = norm_params)
    labels = list(map(myutils.rename_state,sequence.path))
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = n_comp, X = [sequence.sequence],
                              labels = [labels], algorithm = 'labeled')
    return model, sequence.sequence

if __name__ == "__main__":
    args = parsing_json('/home/kirilman/Projects/nir/nir/many_stage/config.json')
    # for k, arg in enumerate(args):
    #     run(arg,k)

    signals = []
    models = []
    for i in range(3):
        if i == 2:
            exp_type = args[i]['type']
            N = args[i]['N']; alpha = args[i]['alpha']; n_comp = args[i]['n_comp']
            norm_params = args[i]['norm_params']
        model , signal = run_test(args[i],i)
        signals +=[signal]
        models += [model]

    args = parsing_json('/home/kirilman/Projects/nir/nir/many_stage/test_sequence.json')
    
    exp_type = args[0]['type']
    N = args[0]['N']; alpha = args[0]['alpha']; n_comp = args[0]['n_comp']
    norm_params = args[0]['norm_params']

    # generator = generator.Sequence(N, alpha, type = exp_type, params=norm_params)
    # test_signal = generator.sequence
    # plt.plot(test_signal)
    # plt.show()
    # f_1 = open('signal.txt', 'w')
    
    with open('result.txt', 'w') as file:
        for arg in args:
            exp_type = arg['type']
            N = arg['N']; alpha = arg['alpha']; n_comp = arg['n_comp']
            norm_params = arg['norm_params']
            for i in range(10):
                gen = generator.Sequence(N, alpha, type = exp_type, params=norm_params)
                test_signal = gen.sequence
                # f_1.write
                signals +=[test_signal]
                log_pr = [model.log_probability(test_signal) for model in models]
                for x in log_pr:
                    file.write(str(x)+'     ')
                file.write(str(np.argmax(log_pr) + 1)+'\n')
            file.write('____\n')
    signals = np.array(signals)
    # signals.tofile('signal.npy')
    np.save('signals.npy',signals)
    # f_1.close()
        # print(np.exp(model.log_probability(test_signal)))
    # print(models)    
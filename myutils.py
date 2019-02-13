import numpy as np
import json
import matplotlib.pyplot as plt
def frequency_occurrence(a,show=True):
    unique = set(a)
    n = len(a)
    d = {}
    for u in unique:
        d.update({u:a.count(u)/n})
    if show == True:
        print(d)
    return d


def table_from_MarkovChain(markov_model):
    obj = markov_model.distributions[1].to_json()
    obj = json.loads(obj)
    return  obj['table']


def get_path_with_viterbi(model):
    path = model.viterbi()
    return path

def rename_state(x):
    a = 's' + str(x)
    return a

def get_y_train(x):
    y = x.copy()
    y = [ el.replace('s','') for el in y]
    y = list(map(int,y))
    return y

def plot_scatter_test_predict(data,y_true,y_pred,dpi = 100, random = False, count = 3):
    data = np.array(data)
    n = data.shape[1]
    plt.style.use('ggplot')
    assert not len(y_pred) != len(y_true),'Размеры массивов y_true и y_pred не совпадают'
    if random == True:
        k = 0
        fig, ax = plt.subplots(count,2,dpi=dpi)
        fig.set_figheight(count*5)
        fig.set_figwidth(count*3)
        m = []
        while k<count:
            i, j = np.random.choice(range(n),2)
            if ([i,j] in m) or ([j,i] in m) or (i == j):
                continue
            else:
                m +=[[i,j]]
                ax[k,0].set_title('Истинная метка')
                ax[k,0].scatter(data[:,i], data[:,j], c = y_true)
                ax[k,0].set_xlabel(str(i))
                ax[k,0].set_ylabel(str(j))

                ax[k,1].set_title('Предсказанная метка')
                ax[k,1].scatter(data[:,i], data[:,j], c = y_pred)
                ax[k,1].set_xlabel(str(i))
                ax[k,1].set_ylabel(str(j))
                k+=1
        fig.tight_layout()
    else:
        fig,ax = plt.subplots(1,2,dpi = dpi)
        ax[0].scatter(data[:,0], data[:,1], c = y_true)
        ax[1].scatter(data[:,0], data[:,1], c = y_pred)
    return


def print_model_distribution(model):
    """
        Формирует строку содержащую матрицу испусканий и матрицу переходов,
        для вывода в файл
    """
    out = '{0:>10}  {1:<15}\n'.format('Состояние', 'Вероятность испускания')
    states = []
    for state in model.states:
        out+='{:>10}: '.format(state.name)
        
        try:
            if len(state.distribution.parameters) > 1:
                print('Количество параметров распределения больше 1, равно ')
                m = state.distribution.parameters[0]
                v = state.distribution.parameters[1]
                temp = "m = {}, v = {} \n".format(m, v)
                out +=temp         
                states += [state.name]
            else:        
                # print('Количество параметров распределения больше 1, равно ', len(state.distibution.parameters))            
                for k,v in state.distribution.parameters[0].items():
                    temp = "'{}' = {:4.2E};    ".format(k,v)
                    out +=temp
                out+='\n'
                states += [state.name]
        except:
            out+='\n'
            
    sample, path = model.sample( 100000, path=True )
    n = model.state_count() - 2
    print(n)
    trans = np.zeros((n,n))
    for state, n_state in zip( path[1:-2], path[2:-1] ):
        state_name = int( state.name[1:] )-1            #!Нестандартное имя состояния
        n_state_name = int( n_state.name[1:] )-1
        trans[ state_name, n_state_name ] += 1
    trans = (trans.T / trans.sum( axis=1 )).T

    out_2 = '        '
    for s in states:
        out_2 += '{:6}'.format(s)  
    out_2+='\n'
    
    for i in range(n):
        for j in range(n):
            if j == 0:
                out_2+='{0:5} {1:4.2E}  '.format(states[i], trans[i,j])
            else:
                out_2+='{:4.2E}  '.format(trans[i,j])
        out_2+='\n'

    out_2 +='     '
    for s in states:
        out_2 += '{:6}'.format(s)
    out_2+='\n'
    for i in range(n):
        for j in range(n):
            if j == 0:
                out_2+='{}  {:.2f}   '.format(states[i], trans[i,j])
            else:
                out_2+='{:.2f}   '.format(trans[i,j])
        out_2+='\n'
            
    out+=out_2
    return  out
# if __name__ == "__main__":
#     a = ['a','b','c','d','b']
#
#     frequency_occurrence(a,False)

def print_trans_matrix(model):
    sample, path = model.sample( 100000, path=True )
    n = model.state_count() - 2
    trans = np.zeros((n,n))
    for state, n_state in zip( path[1:-2], path[2:-1] ):
        state_name = int( state.name[1:] )-1            #!Нестандартное имя состояния
        n_state_name = int( n_state.name[1:] )-1
        trans[ state_name, n_state_name ] += 1
    
    trans = (trans.T / trans.sum( axis=1 )).T
    
    print(trans) 



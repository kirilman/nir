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
    a = 's'+x
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
# if __name__ == "__main__":
#     a = ['a','b','c','d','b']
#
#     frequency_occurrence(a,False)



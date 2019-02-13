import matplotlib.pyplot as plt
import numpy.random as random
from multiprocessing import Pool

def test_plt(a,b,number):
    fig_loc = plt.figure(num = number + 1, )
    plt.scatter(a,b )
    plt.savefig("%043d.jpg" % (number + 1,))

    plt.close()

def do_plot(number):
    fig = plt.figure(number)

    a = random.sample(1000)
    b = random.sample(1000)

    # generate random data
    plt.scatter(a, b)

    plt.savefig("%03d.jpg" % (number,))
    plt.close()

    test_plt(a,b, number + 1)
    print("Done ", number)


if __name__ == '__main__':
    pool = Pool()
    pool.map(do_plot, range(3))
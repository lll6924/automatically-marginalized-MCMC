import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
setting = ['PulmonaryFibrosisVectorized/_10000_100000/HMC','PulmonaryFibrosisReparameterized/_10000_100000/HMC','PulmonaryFibrosis/_m_a-s_a-m_b-s_b_10000_100000/MHMC']
label = ['HMC-V','HMC-R','HMC-M']
keys = [0,1,2,3,4]
n = 3

def main():
    path = 'result/'
    for i in range(n):
        nums = []
        times = []
        esss = []
        for key in keys:
            file = path + f'{setting[i]}/{key}'
            with open(file) as f:
                lines = f.readlines()
                num = float(lines[-1])
                if i==0 or i == 1:
                    time = float(lines[0])
                else:
                    time = float(lines[0])+float(lines[1])
            times.append(time/60)
            nums.append(num/60)
            esss.append(num*time)
        print(label[i], "%0.1f" % np.mean(esss), "(%0.1f)" % np.std(esss),'&', "%0.1f" % np.mean(times), "(%0.1f)" % np.std(times),'&',"%0.1f" % np.mean(nums), "(%0.1f)" % np.std(nums))




if __name__ == '__main__':
    main()

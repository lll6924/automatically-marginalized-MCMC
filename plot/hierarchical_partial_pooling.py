import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

model = ['baseball_small','baseball_large','rat_tumors']
setting = ['_10000_100000/HMC','__10000_100000/MHMC']
label = ['HMC','HMC-M']
keys = [0,1,2,3,4]
n = 2

def main():
    path = 'result/'
    for m in model:
        for i in range(n):
            nums = []
            times = []
            esss = []
            for key in keys:
                file = path + f'HierarchicalPartialPooling/{m}{setting[i]}/{key}'
                with open(file) as f:
                    lines = f.readlines()
                    num = float(lines[-1])
                    if i==0:
                        time = float(lines[0])
                    else:
                        time = float(lines[0])+float(lines[1])
                times.append(time)
                nums.append(num)
                esss.append(num*time)
            print(m, label[i], "%0.1f" % np.mean(esss), "(%0.1f)" % np.std(esss),'&', "%0.1f" % np.mean(times), "(%0.1f)" % np.std(times),'&',"%0.1f" % np.mean(nums), "(%0.1f)" % np.std(nums))




if __name__ == '__main__':
    main()

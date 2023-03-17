import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

model = ['baseball_small','baseball_large','rat_tumors']
keys = [0,1,2,3,4]

def main():
    path = 'result/StanHPP/'
    for m in model:
            nums = []
            times = []
            esss = []
            for key in keys:
                file = path + m + '/' + str(key)
                if not os.path.exists(file):
                    continue
                with open(file) as f:
                    line = f.readlines()[0].split(' ')
                times.append(float(line[1]))
                nums.append(float(line[2]))
                esss.append(float(line[0]))
            print(m,  "%0.1f" % np.mean(esss), "(%0.1f)" % np.std(esss),'&', "%0.1f" % np.mean(times), "(%0.1f)" % np.std(times),'&',"%0.1f" % np.mean(nums), "(%0.1f)" % np.std(nums))




if __name__ == '__main__':
    main()

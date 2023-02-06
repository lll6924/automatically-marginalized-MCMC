import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

font = {
        'size'   : 22}

matplotlib.rc('font', **font)

model = ['ElectricCompany','ElectricCompanyReparameterized','ElectricCompany']
setting = ['_10000_100000/HMC','_10000_100000/HMC','_mua0-mua1-mua2-mua3_10000_100000/MHMC']
label = ['HMC','HMC-R','HMC-M']
keys = [0,1,2,3,4]
colors = ['b','g','r']
n = 3

def main():
    path = 'result/'
    data = [[] for _ in range(n)]
    for i in range(n):
        numberss = []
        for key in keys:
            file = path + f'{model[i]}/{setting[i]}/{key}'
            with open(file) as f:
                lines = f.readlines()
                if i==2:
                    s = float(lines[0])+float(lines[1])
                    lines = lines[2:-2]
                else:
                    s = float(lines[0])
                    lines = lines[1:-2]
            numbers = []
            for line in lines:
                line = line.split()
                if line[-1] == ']':
                    line = line[:-1]
                line = [float(l.replace('[','').replace(']','').replace(',','').replace('\n',''))/s for l in line]
                numbers.extend(line)
            numbers = sorted(numbers)
            for j in range(len(numbers)):
                data[i].append({'x': j, 'ESS/s': numbers[j]})


    f, ax = plt.subplots(figsize=(8,4))
    sns.set_color_codes("muted")

    for i in reversed(range(n)):
        d = pd.DataFrame(data[i])
        sns.lineplot(x="x", y="ESS/s", data=d,
                    label=label[i], color=colors[i],ax=ax)
    ax.legend(ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.45, 0.95))
    ax.set_yscale("log")
    ax.set_xlabel('component')
    #ax.set_xticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

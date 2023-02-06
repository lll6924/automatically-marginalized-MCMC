import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

font = {
        'size'   : 22}

matplotlib.rc('font', **font)

settingss = [[50,100,150,200,250],[100,200,300,400,500]]
files = ['0_compile','0_no_marginalization_compile']
titles = ['Marginalized model', 'Original model']
def main():
    path = 'result/TMP/'
    data1 = []

    for title, settings,file_name in zip(titles,settingss,files):
        x = []
        y1 = []
        y2 = []
        for setting in settings:
            file = path + f'{setting}__10000_100000/MHMC/{file_name}'
            print(file)
            if os.path.exists(file):
                with open(file, 'r') as f:
                    line = f.readline().split()
                    size1 = float(line[0])
                    size2 = float(line[1])
                    time = float(line[2])
                    x.append(setting)
                    y1.append(size2)
                    y2.append(time)
                    data1.append({'N':setting,'Lines':size1,'Time':time})



        fig, ax1 = plt.subplots(figsize=(8,5))

        ax2 = ax1.twinx()
        ax1.plot(x, y1, 'g-')
        ax2.plot(x, y2, 'b-')
        ax1.set_ylim([1,25000])
        ax1.set_title(title)
        ax2.set_ylim([1,np.max(y2)*1.2])
        ax1.set_xlim([0,settings[-1]+50])
        ax1.set_xlabel('N')
        ax1.set_ylabel('Lines of Jaxprs', color='g')
        ax2.set_ylabel('Compilation time (s)', color='b')
        plt.tight_layout()
        plt.show()

    # sns.set_color_codes("muted")
    #
    # data1 = pd.DataFrame(data1)
    # print(data1)
    #
    # sns.lineplot(data=data1,x='N',y='Lines')
    # plt.show()
    # plt.clf()
    # sns.lineplot(data=data1,x='N',y='Time')
    # plt.show()
    # plt.clf()

if __name__ == '__main__':
    main()

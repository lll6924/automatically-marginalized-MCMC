import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
sns.set_context("notebook", font_scale=2)
f, ax = plt.subplots(figsize=(8, 6))

sample_file_1 = 'result/PulmonaryFibrosisVectorized/1400_10000_100000/HMC/0.npz'

sample_file_2 = 'result/PulmonaryFibrosis/1400_m_a-s_a-m_b-s_b_10000_100000/MHMC/0.npz'

data = []
f1 = np.load(sample_file_1)['s_b'][0]
f2 = np.load(sample_file_2)['s_b'][0,:,0]
for i in range(len(f1)):
    data.append({'$\\log\\sigma_b$':np.log(f1[i]),'algorithm':'HMC-V'})
    data.append({'$\\log\\sigma_b$': np.log(f2[i]), 'algorithm': 'HMC-M'})
data = pd.DataFrame(data)
ax = sns.histplot(data=data,x='$\\log\\sigma_b$',hue='algorithm',stat='density',kde=True,bins=40)
sns.move_legend(ax, "upper left")
plt.tight_layout()
plt.show()

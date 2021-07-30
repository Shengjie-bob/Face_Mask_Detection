import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用


# Plot 超参数 图
path = './serverdata/noise'
hyparameter ='noise'

hyp_files =glob.glob('{}/{}.txt'.format(path,hyparameter))

mAPs =[];AP0s=[];AP1s =[];
for file in hyp_files:
    lines = open(file).readlines()
    i =0
    for line in lines:
        line =line.split(' ')
        mAP = float(line[5])
        if i % 3 == 0:
            mAPs.append(mAP)
        elif i % 3 == 1:
            AP0s.append(mAP)
        elif i % 3 == 2:
            AP1s.append(mAP)
        i = i + 1
mAPs =np.array(mAPs).astype(np.float);AP0s= np.array(AP0s).astype(np.float); AP1s =np.array(AP1s).astype(np.float)
index =np.argsort(mAPs)
mAPs =mAPs[index]
AP0s =AP0s[index]
AP1s = AP1s[index]

hyp_name ='noise'

hyps = np.array([0.94,0.98,1])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(hyps, mAPs,'o-',c='darkorchid')
ax.plot(hyps, AP0s,'*-',c='cyan')
ax.plot(hyps, AP1s,'D-',c='salmon')
ax.set_xlabel('SNR')
ax.set_ylabel('measurement')
plt.legend(['ALL-mAP@[.5:.95]','Face-mAP@[.5:.95]','Facemask-mAP@[.5:.95]'],loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('{}/{}.png'.format(path,hyp_name), dpi=300)
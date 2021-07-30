import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用


# Plot 超参数 图
path = './serverdata/experiments'
hyparameter ='soft_nms'

hyp_files =glob.glob('{}/*{}*.txt'.format(path,hyparameter))

hyps =[];mAPs=[];f1s =[];
for file in hyp_files:
    lines = open(file).readlines()
    for line in lines:
        line =line.split(' ')
        hyp =float(line[0])
        mAP = float(line[3])
        f1 = float(line[-1])
        hyps.append(hyp)
        mAPs.append(mAP)
        f1s.append(f1)
hyps =np.array(hyps).astype(np.float);mAPs= np.array(mAPs).astype(np.float); f1s =np.array(f1s).astype(np.float)
index =np.argsort(hyps)
hyps =hyps[index]
mAPs =mAPs[index]
f1s = f1s[index]

hyp_name ='soft'

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(hyps, mAPs,'o-',c='darkorchid')
# ax.plot(hyps, f1s,'*-',c='cyan')
ax.set_xlabel('$\sigma$_{}'.format(hyp_name))
ax.set_ylabel('measurement')
plt.legend(['mAP@[.5:.95]',],loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('{}/{}.png'.format(path,hyparameter), dpi=300)
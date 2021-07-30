import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用


# Plot 超参数 图
path = './serverdata/nms'
hyparameter ='soft'

hyp_files =glob.glob('{}/*{}*.txt'.format(path,hyparameter))

hyps =[];mAPs=[];aps0=[];aps1=[]
for file in hyp_files:
    name=file.split('_')
    hyp = float(name[-1][:-4])
    lines = open(file).readlines()
    for line in lines:
        line =line.split(' ')
        mAP = float(line[1])
        if line[0]=='face:':
            aps0.append(mAP)
        if line[0]=='face_mask:' :
            aps1.append(mAP)
    hyps.append(hyp)
    mAPs.append(mAP)
hyps =np.array(hyps).astype(np.float);mAPs= np.array(mAPs).astype(np.float)
aps0 =np.array(aps0).astype(np.float);aps1 =np.array(aps1).astype(np.float)
index =np.argsort(hyps)
hyps =hyps[index];mAPs =mAPs[index];aps0=aps0[index];aps1=aps1[index]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(hyps, mAPs,'D-',c='salmon')
ax.plot(hyps, aps0,'o-',c='darkorchid')
ax.plot(hyps, aps1,'*-',c='limegreen')
ax.set_xlabel('$\sigma$_{}'.format(hyparameter))
ax.set_ylabel('measurement')
plt.legend(['mAP@[.5:.95]','face-AP@[.5:.95]','facemask-AP@[.5:.95]'],loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('{}/{}.png'.format(path,hyparameter), dpi=300)
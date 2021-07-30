import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用


# Plot 超参数 图
path = './hyp'
hyparameter ='giou'

hyp_files =glob.glob('{}/*{}*.txt'.format(path,hyparameter))

ps=[];rs=[];maps=[];f1s=[]
for file in hyp_files:
    lines = open(file).readlines()
    for line in lines:
        line =line.split()
        P = line[8]
        R = line[9]
        mAP = line[10]
        F1 = line[11]
    ps.append(P)
    rs.append(R)
    maps.append(mAP)
    f1s.append(F1)
ps =np.array(ps).astype(np.float);rs= np.array(rs).astype(np.float)
maps =np.array(maps).astype(np.float);f1s =np.array(f1s).astype(np.float)

if hyparameter =='iou_t':

    hyp = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

if hyparameter =='cls':

    hyp = np.array([10,20,30,40,50,60,70,80])

if hyparameter =='obj':

    hyp = np.array([10,20,30,40,50,60,70,80])

if hyparameter =='giou':

    hyp = np.array([1,2,3,4,5,6,7,8])


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(hyp, ps,'o-',c='salmon')
ax.plot(hyp, rs,'*-',c='limegreen')
ax.plot(hyp, maps,'^-',c='cyan')
ax.plot(hyp, f1s,'v-',c='darkorchid')
ax.set_xlabel('$\lambda$_{}'.format(hyparameter))
ax.set_ylabel('measurement')
plt.legend(['Precision','Recall','mAP','F1'],loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('{}/{}.png'.format(path,hyparameter), dpi=300)
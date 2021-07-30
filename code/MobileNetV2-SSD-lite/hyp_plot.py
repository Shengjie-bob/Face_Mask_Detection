import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用


# Plot 超参数 图
path = './eval_results'
hyparameter ='ratio'

hyp_files =glob.glob('{}/{}.txt'.format(path,hyparameter))

hyps =[];regs=[];clcs=[]
for file in hyp_files:
    lines = open(file).readlines()
    for line in lines:
        line =line.split()
        hyp = line[3]
        reg = line[7]
        clc = line[9]
        hyps.append(hyp)
        regs.append(reg)
        clcs.append(clc)
hyps =np.array(hyps).astype(np.float);regs= np.array(regs).astype(np.float)
clcs =np.array(clcs).astype(np.float)

if hyparameter == 'iou':
    hyp_name ='$\lambda$_iou_t'
elif hyparameter =='alpha':
    hyp_name =r'$\alpha$'
elif hyparameter =='ratio':
    hyp_name ='$\gamma$'



fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(hyps, regs,'o-',c='salmon')
ax.plot(hyps, clcs,'*-',c='limegreen')
# ax.plot(hyps, maps,'^-',c='cyan')
# ax.plot(hyps, f1s,'v-',c='darkorchid')
ax.set_xlabel('{}'.format(hyp_name))
ax.set_ylabel('measurement')
plt.legend(['Regressionloss','Classifierloss'],loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('{}/{}.png'.format(path,hyparameter), dpi=300)
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格
sns.set_palette("muted") #常用



# Plot PR 图
path = './eval_results'
thresh = 0.9
recall0 = np.load('{}/recall_1_{}.npy'.format(path,thresh))
precision0 = np.load('{}/precision_1_{}.npy'.format(path,thresh))

recall1 = np.load('{}/recall_2_{}.npy'.format(path,thresh))
precision1 = np.load('{}/precision_2_{}.npy'.format(path,thresh))

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(recall0, precision0,'-',c='salmon')
ax.plot(recall1, precision1,'-',c='darkorchid')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.legend(['face','face_mask'],loc=[0.7,1])
plt.title('iou_thresh={}'.format(thresh))
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.01)
fig.tight_layout()
plt.show()
fig.savefig('{}/PR_curve_{}.png'.format(path,thresh), dpi=300)
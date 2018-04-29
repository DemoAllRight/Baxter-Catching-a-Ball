from data_manager import data_manager
from cnn import CNN
from trainer import Solver
# from viz_features import Viz_Feat
import random


import matplotlib.pyplot as plt

output_size = 7
feature_size = 12

random.seed(0)

dm = data_manager('data/')

cnn = CNN(output_size,feature_size)

solver = Solver(cnn,dm)

solver.optimize()

plt.plot(solver.test_loss,label = 'Validation')
plt.plot(solver.train_loss, label = 'Training')
plt.legend()
plt.xlabel('Iterations (in 200s)')
plt.ylabel('loss')
plt.show()

# val_data = dm.val_data
# train_data = dm.train_data

# sess = solver.sess

# cm = Viz_Feat(val_data,train_data,CLASS_LABELS,sess)

# cm.vizualize_features(cnn)





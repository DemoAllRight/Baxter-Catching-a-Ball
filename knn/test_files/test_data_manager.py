import sys
sys.path.append('../')

from data_manager import *

dm = data_manager('../data/')

train_features , train_labels = dm.get_train_batch()
test_features , test_labels = dm.get_validation_batch()

print('train features shape:' , train_features.shape)
print('train labels shape:' , train_labels.shape)
print('test features shape:' , test_features.shape)
print('test labels shape:' , test_labels.shape)
print('\n\n')
print('one example train feature',train_features[0])
print('\n\n')
print('one example train label',train_labels[0])
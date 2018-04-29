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
print(train_features)
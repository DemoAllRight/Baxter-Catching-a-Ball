import sys
sys.path.append('../')

from data_manager import *

from sklearn.neighbors import KNeighborsRegressor

dm = data_manager('../data/')
dm.batch_size = len(dm.train_data)
dm.val_batch_size = len(dm.val_data)

train_features , train_labels = dm.get_train_batch()
test_features , test_labels = dm.get_validation_batch()

# print('train features shape:' , train_features.shape)
# print('train labels shape:' , train_labels.shape)
# print('test features shape:' , test_features.shape)
# print('test labels shape:' , test_labels.shape)
# print('\n\n')
# print(train_features)

neigh = KNeighborsRegressor(n_neighbors=5)

neigh.fit(train_features,train_labels)

prediction = neigh.predict(test_features)

error = np.mean( np.square( prediction - test_labels ) )

print(error)

print(prediction[0])
print(test_labels[0])
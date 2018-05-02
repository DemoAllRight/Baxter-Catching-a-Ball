import sys
sys.path.append('../')

from data_manager import *

from sklearn.neighbors import KNeighborsRegressor

dm = data_manager('../data/')
# dm.batch_size = len(dm.train_data)
# dm.val_batch_size = len(dm.val_data)

train_features , train_labels = dm.get_train_batch()
test_features , test_labels = dm.get_validation_batch()

# print('train features shape:' , train_features.shape)
# print('train labels shape:' , train_labels.shape)
# print('test features shape:' , test_features.shape)
# print('test labels shape:' , test_labels.shape)
# print('\n\n')
# print(train_features)

for i in range(20):

	neigh = KNeighborsRegressor(n_neighbors=i+1)

	neigh.fit(train_features,train_labels)

	prediction = neigh.predict(test_features)

	error = np.sqrt(np.mean( np.square( prediction - test_labels ) ))

	print('{} nearest neighbors training mean square root error:'.format(i+1),error)

# print(prediction[20])
# print(test_labels[20])
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import linear_model
import pickle

data = np.load("/home/baxter2/Desktop/causal_confusion/custom/Qlearning_MountainCar/expert_data_2019_10_30_16_20.npy")

np.random.shuffle(data)
np.random.shuffle(data)

"""With confounding variable"""
X1 = data[:,0:3]
y = data[:,3]

G = np.random.binomial(1,0.5,[X1.shape[0],X1.shape[1]])
X2 = np.multiply(X1,G)
X_G = np.concatenate((X2,G),axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_G, y, test_size=0.20, random_state = 0)

# X_train = X
# y_train = y
# lr = linear_model.LogisticRegression()
# lr.fit(X_train, y_train)

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', verbose=1).fit(X_train, y_train)

print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, mul_lr.predict(X_test)))

# save the model to disk
filename = 'mul_lr_model.sav'
pickle.dump(mul_lr, open(filename, 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

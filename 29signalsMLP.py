import numpy as np
import keras
import itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
#os.environ["KERAS_BACKEND"] = "theano"
#os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
#import theano as th
#import theano.tensor as T
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
from keras import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import pandas as pd
import pickle


def getData(data1):

    X = data1[:,:-2].copy()
    y = data1[:,-2].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
        
    return X_train, X_test, y_train, y_test
    

with (open('/Users/donna/Desktop/VanhoyProject/mod_29_rsf.p', 'rb')) as f:
    data = pickle.load(f, encoding='latin1')         # load file content as mydict
    f.close()                       


data = np.asarray(data)
X_train, X_test, y_train, y_test = getData(data)



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Ytst = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(encoded_Ytst)

def plot_confusion_matrix(cm, classes, normalize=True, title= 'Confusion Matrix', cmap=plt.cm.GnBu):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = ' .2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# create model
model = Sequential()
model.add(Dense(1000, input_dim=256, activation='relu',bias_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(1000, input_dim=1000, activation='relu',bias_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(1050, input_dim=1000, activation='relu',bias_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(1050, input_dim=1050, activation='relu',bias_regularizer=keras.regularizers.l2(0.01)))

model.add(Dense(29, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=256, validation_split = .2, verbose =2)
score = model.evaluate(X_test, y_test, batch_size=64)
print(score)


class_names = ['8qam_circular', 'am-dsb', '8cpfsk', 'lfm_squarewave', '8pam', 'ofdm-64-bpsk', 'lfm_sawtooth', '8gfsk', '16qam', 'ofdm-16-bpsk', '32qam_rect', '4ask', '16psk', 
               'am-ssb', '2gfsk', 'ofdm-32-bpsk', '2cpfsk', '4cpfsk', '64qam', '4pam', 'ofdm-64-qpsk', '4gfsk', 'ook', '32qam_cross', '8qam_cross', 'ofdm-32-qpsk', 'ofdm-16-qpsk', 'wbfm', 'bpsk']


# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=64)
conf = np.zeros([len(class_names),len(class_names)])
confnorm = np.zeros([len(class_names),len(class_names)])
for i in range(0,X_test.shape[0]):
    j = list(y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(class_names)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, classes=class_names, normalize = True,
                      title='Normalized Confusion Matrix')


# serialize model to JSON

model_json = model.to_json()
#with open("estimator.json", "w") as json_file:
#    json_file.write(estimator_json)
# serialize weights to HDF5
model.save("modulation_MLP.h5")
print("Saved model to disk")

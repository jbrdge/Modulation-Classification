import numpy as np
import pandas
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
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


dataframe = pandas.read_csv("/Volumes/LPR_2018/modulation_classification/MATLAB/128samples-Train.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:256].astype(float)
Y = dataset[:,256]

dataframe = pandas.read_csv("/Volumes/LPR_2018/modulation_classification/MATLAB/128samples-Test.csv", header=None)
dataset = dataframe.values
Xtst = dataset[:,0:256].astype(float)
Ytst = dataset[:,256]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
y_trn = np_utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(Ytst)
encoded_Ytst = encoder.transform(Ytst)
# convert integers to dummy variables (i.e. one hot encoded)
y_tst = np_utils.to_categorical(encoded_Ytst)

def plot_confusion_matrix(cm, classes, normalize=False, title= 'Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting 'normalize=True'
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, without normalization')

    print(cm)
    
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
model.add(Dense(1000, input_dim=256, activation='relu',bias_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1000, input_dim=1000, activation='relu',bias_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1050, input_dim=2000, activation='relu',bias_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1050, input_dim=1050, activation='relu',bias_regularizer=keras.regularizers.l2(0.001)))

model.add(Dense(9, activation='softmax'))
# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='Adamax',
              metrics=['accuracy'])

model.fit(X, y_trn,
          epochs=1,
          batch_size=64, validation_split = .2, verbose =2)
score = model.evaluate(Xtst, y_tst, batch_size=64)
print(score)


class_names = ['BPSK', 'QPSK', '8PSK', '16QAM', 'CPFSK', 'GMSK', 'FM', 'AM', 'OFDM']
# Plot confusion matrix
test_Y_hat = model.predict(Xtst, batch_size=64)
conf = np.zeros([len(class_names),len(class_names)])
confnorm = np.zeros([len(class_names),len(class_names)])
for i in range(0,Xtst.shape[0]):
    j = list(y_tst[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(class_names)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, classes=class_names, normalize = True,
                      title='Normalized confusion matrix')


#plot_confusion_matrix(confnorm, labels=class_names)




'''
y_pred = model.predict(Xtst)
print(y_tst)
print(y_pred)
print(y_pred.shape)

#compute confusion matrix
print(classification_report(y_tst, y_pred, target_names=class_names))
print(confusion_matrix(y_tst, y_pred, labels=range(9)))
np.set_printoptions(precision=2)
'''
'''
#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix without normalization')

#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = True,
                      title='Normalized confusion matrix')

plt.show()
'''


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#return model






#estimator = KerasClassifier(model, epochs=1, batch_size=64, verbose=1)
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# serialize model to JSON

model_json = model.to_json()
#with open("estimator.json", "w") as json_file:
#    json_file.write(estimator_json)
# serialize weights to HDF5
model.save("modulation_MLP.h5")
print("Saved model to disk")

 
# later...

'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''











'''
trn_data = np.genfromtxt("/Volumes/LPR_2018/data/Train.csv", delimiter=',')
tst_data = np.genfromtxt("/Volumes/LPR_2018/data/Test.csv", delimiter=',')
X_trn = trn_data[:,0:-1]
y_trn = trn_data[:,-1]
X_tst = tst_data[:,0:-1]
y_tst = tst_data[:,-1]

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=800))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_trn, y_trn, epochs=10, batch_size=100, verbose=0)
print('50HLN')
train_score = model.evaluate(X_trn,y_trn, verbose=0)
print('Training Error:', 1-train_score[1])
score = model.evaluate(X_tst, y_tst, verbose=0)
print('Test Error:', 1-score[1])
'''
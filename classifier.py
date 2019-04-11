''' This Neural Network predicts Pipe Burst'''
'''Author:Developer Prince'''

import tensorflow as tf
from tensorflow.keras import layers # keras for DNN
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


os.getcwd()

os.listdir(os.getcwd())

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)

def __obtain_data__(path, number_features, number_labels):
    data_set = pd.read_csv(path, low_memory = False)
    class_columns = number_labels + number_features
    input_x = data_set.iloc[ : , 1 : number_features]
    input_y = data_set.iloc[ : , number_features : class_columns]
    x_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = 0)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = __obtain_data__(path='data/pumula_metrics.csv', number_features=2, number_labels=1)

model = tf.keras.Sequential() # sequential model(feed forward neural network)
model.add(layers.Dense(4, activation='relu'))   # number of nodes in first layer ,relu to activate this layer,reduced the number of nuerons from 64 to 17
model.add(layers.Dense(8, activation='relu'))   # number of nodes in first layer ,relu to activate this layer
model.add(layers.Dense(2, activation='sigmoid'))
model.add(layers.Dense(1, bias_initializer=tf.keras.initializers.constant(1.0)))    # creates a forceful bias ,to minimise the losses

model.compile(optimizer=tf.train.AdamOptimizer(0.02),   # optimisation functioon
              loss='mse',  
              metrics=['accuracy'])


history = model.fit(x=x_train.values, y=y_train.values, epochs=1000, validation_data=(x_test, y_test))    # trains the model and returns metrics of the loss at each epoch(iteration)
model.save('models/pipeburts.h5')
success = "model generation successful"

model.summary() # gives a summary of what the model is built from


CA = tf.keras.models.load_model('models/pipeburts.h5')      # import model and test to see if it can predict against test values
predictions = CA.predict_classes(x=x_test.values).flatten()
print(predictions)
plt.title('Burst Pipe Prediction Model Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
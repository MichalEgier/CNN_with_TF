import tensorflow as tf
#for general network models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, add, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
#for dataset
from keras.datasets import mnist
#for convolutional network
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D
#for plots
import matplotlib.pyplot as plt
#for time measuring
import time

#load data from dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

def draw_plot(history, ylim=(0.5, 1.00)):
    plt.figure(figsize=(15, 5))
    plt.plot(history.history['accuracy'], "y--")
    plt.plot(history.history['val_accuracy'], "r--")
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.ylim(ylim)
    plt.legend(['train', 'test'], loc='best')

    plt.show()


network_MTL = Sequential()
network_MTL.add(Flatten(input_shape=(28,28)))
network_MTL.add(Dense(100, activation="relu"))
network_MTL.add(Dense(10, activation="softmax"))

network_MTL.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

network_MTL.summary()

start_time_1 = time.time()

history = network_MTL.fit(train_X, train_y,
                    epochs=10, verbose=1,
                    validation_data=(test_X, test_y),
                    )

end_time_1 = time.time()

draw_plot(history)

score = network_MTL.evaluate(test_X, test_y, verbose=0)
print("MTL Error: %.2f%%" % (100 - score[1] * 100))


#Convolutional network


#added one dimension for canal
X_train_cnn = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X_cnn = test_X.reshape((test_X.shape[0], 28, 28, 1))
#this won't work
#X_train_cnn = train_X
#test_X_cnn = test_X

network_Conv = Sequential()
network_Conv.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#network_Conv.add(MaxPool2D(pool_size=(2, 2)))  #not yet - checked in next model

network_Conv.add(Flatten())
network_Conv.add(Dense(100, activation='relu'))
network_Conv.add(Dense(10, activation='softmax'))

network_Conv.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

network_Conv.summary()

start_time_2 = time.time()

history = network_Conv.fit(X_train_cnn,
                    train_y,
                    epochs=10,
                    verbose=1,
                    validation_data=(test_X_cnn, test_y),
                    )

end_time_2 = time.time()

draw_plot(history)

score = network_Conv.evaluate(test_X_cnn, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - score[1] * 100))



start_times = []
end_times = []

#various convolutional networks with poolings
for pooling_layer in [MaxPool2D(pool_size=(2,2)), AvgPool2D(pool_size=(2,2)), MaxPool2D(pool_size=(3,3))]:
    #Convolutional network with pooling (2,2) MAX

    #the same like before
    X_train_cnn = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X_cnn = test_X.reshape((test_X.shape[0], 28, 28, 1))

    network_Conv_Pooling = Sequential()
    network_Conv_Pooling.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network_Conv_Pooling.add(pooling_layer)   #this time we're trying with pooling

    network_Conv_Pooling.add(Flatten())
    network_Conv_Pooling.add(Dense(100, activation='relu'))
    network_Conv_Pooling.add(Dense(10, activation='softmax'))

    network_Conv_Pooling.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    network_Conv_Pooling.summary()

    start_times.append(time.time())

    history = network_Conv_Pooling.fit(X_train_cnn,
                        train_y,
                        epochs=10,
                        verbose=1,
                        validation_data=(test_X_cnn, test_y),
                        )

    end_times.append(time.time())

    draw_plot(history)

    score = network_Conv_Pooling.evaluate(test_X_cnn, test_y, verbose=0)
    print("CNN with pooling " + str(pooling_layer) + " Error: %.2f%%" % (100 - score[1] * 100))



print("Times:")
print("MLP: " + str(end_time_1 - start_time_1))
print("CNN: " + str(end_time_2 - start_time_2))
print("CNN with pooling MAX (2,2): " + str(end_times[0] - start_times[0]))
print("CNN with pooling AVG (2,2): " + str(end_times[1] - start_times[1]))
print("CNN with pooling MAX (3,3): " + str(end_times[2] - start_times[2]))

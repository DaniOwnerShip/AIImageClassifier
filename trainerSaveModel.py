import tensorflow as tf 
import numpy as np
import pandas as pd   
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle 
from sklearn.model_selection import train_test_split


#sasha/dog-food
data = pd.read_parquet('train-00000-of-00001-9bf5abf8b080cbba.parquet')

# Dividir los datos en entrenamiento y validación
imagesTrain, imagesVal, labelTrain, labelVal = train_test_split(data['image'], data['label'], test_size=0.2, random_state=42)

# Convertir las listas en arrays numpy
imagesTrain = np.array(imagesTrain.tolist())
labelTrain = np.array(labelTrain.tolist())
imagesVal = np.array(imagesVal.tolist())
labelVal = np.array(labelVal.tolist())

# image preprocessing 
image_tensorsTrain = []
for image in imagesTrain:
    image_bytes = image['bytes']
    image_tensor = tf.image.decode_jpeg(image_bytes) 
    resized_image = tf.image.resize(image_tensor, (128, 128))
    resized_image = tf.image.rgb_to_grayscale(resized_image) 
    image_tensorsTrain.append(resized_image)  
image_tensorsTrain = np.array(image_tensorsTrain)  

image_tensorsTest = []
for image in imagesVal:
    image_bytes = image['bytes']
    image_tensor = tf.image.decode_jpeg(image_bytes) 
    resized_image = tf.image.resize(image_tensor, (128, 128))
    resized_image = tf.image.rgb_to_grayscale(resized_image) 
    image_tensorsTest.append(resized_image)  
image_tensorsTest = np.array(image_tensorsTest)  


# keras model instance 
model = Sequential()

# architecture layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# save log function for TensorBoard app
# launch command: tensorboard --logdir=logs/
tensorboard_callback = TensorBoard(log_dir="./logs") 

# Train model
history = model.fit(image_tensorsTrain, labelTrain,
                    epochs= 10,
                    batch_size=16,
                    validation_data=(image_tensorsTest, labelVal),
                    callbacks=[tensorboard_callback]) 


# evaluate model 
test_loss, test_acc = model.evaluate(image_tensorsTest, labelVal, verbose=2)

# print accuracy
print('Test accuracy:', test_acc)


# Save trained model 
with open('trained_model-10-16.pkl', 'wb') as file:
    pickle.dump(model, file)





 
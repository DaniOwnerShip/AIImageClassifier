import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd 
from PIL import Image 
from sklearn.model_selection import train_test_split  
from keras import layers 
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import io
  
dataTest = pd.read_parquet('test-00000-of-00001-6ea6ccdcc8fa38d5.parquet')
imagesTest = np.array(dataTest['image'].tolist())  
labelTest = np.array(dataTest['label'].tolist())       

dataTrain = pd.read_parquet('train-00000-of-00001-9bf5abf8b080cbba.parquet') 
imagesTrain = np.array(dataTrain['image'].tolist())  
labelTrain = np.array(dataTrain['label'].tolist())  


print('tipe ', len(imagesTest))
print('tipe ', type(imagesTest[880])) 
 

for i in np.arange(0, 51):
    img_bytes = imagesTest[i]['bytes']
    img_data = io.BytesIO(img_bytes) 
    img = Image.open(img_data)  
    image = img.convert("L") 
    image = image.resize((128, 128)) 
    image.save(f'./ImgTestHuman2/imageTest{i}.jpg')
#img.img_bytes('imageTest.jpg')

#print('tipe ', imagesTest[880]['bytes'])

print('**************')
img_bytes = imagesTest[0]['bytes']
immg = tf.image.decode_jpeg(img_bytes) 
resized_image = tf.image.resize(immg, (128, 128))
resized_image = tf.image.rgb_to_grayscale(resized_image) 
print('sss ', resized_image.shape)

#imgg = image_tensorsTrain[9,:,:]  



image_tensorsTrain = []
for image in imagesTrain:
    image_bytes = image['bytes']
    image_tensor = tf.image.decode_jpeg(image_bytes) 
    resized_image = tf.image.resize(image_tensor, (128, 128))
    resized_image = tf.image.rgb_to_grayscale(resized_image) 
    image_tensorsTrain.append(resized_image)  
image_tensorsTrain = np.array(image_tensorsTrain)

 

image_tensorsTest = []
for image in imagesTest:
    image_bytes = image['bytes']
    image_tensor = tf.image.decode_jpeg(image_bytes) 
    resized_image = tf.image.resize(image_tensor, (128, 128))
    resized_image = tf.image.rgb_to_grayscale(resized_image) 
    image_tensorsTest.append(resized_image)  
image_tensorsTest = np.array(image_tensorsTest)

print('tipe ', image_tensorsTest.dtype)


"""
img = Image.fromarray(image_tensorsTest[0].astype('uint8')) 
img.save('imageTest.png')
img_data = io.BytesIO(img_bytes) 
img = Image.open(img_data)  
image = img.convert("L") 
image = image.resize((128, 128)) 
image.save('imageTest.jpg')
gpu_available = tf.test.is_built_with_cuda()
print("gpu_available " , gpu_available)
"""

tensorboard_callback = TensorBoard(log_dir="./logs")

print("image_tensorsTrain ", len(image_tensorsTrain))  


fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.ravel()
for i in np.arange(839, 848):
    axes[i-839].imshow(image_tensorsTrain[i]) 
    axes[i-839].set_title(labelTrain[i])
    axes[i-839].axis('off')
    plt.subplots_adjust(wspace=1)
plt.show()



print("image_tensorsTrain ", len(image_tensorsTrain)) 

# Define the model architecture
model = Sequential()
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

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(image_tensorsTrain, labelTrain,
                    epochs=10,
                    batch_size=64,
                    validation_data=(image_tensorsTest, labelTest),
                    callbacks=[tensorboard_callback])


# evaluar modelo en conjunto de datos de prueba
test_loss, test_acc = model.evaluate(image_tensorsTest, labelTest, verbose=2)

print('Test accuracy:', test_acc)



"""
print("image_tensorsTrain ", len(image_tensorsTrain)) 
# mostrar imágenes
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.ravel()
for i in np.arange(839, 848):
    axes[i-839].imshow(image_tensorsTrain[i])
    axes[i-839].set_title(labelTrain[i])
    axes[i-839].axis('off')
    plt.subplots_adjust(wspace=1)
plt.show()
"""
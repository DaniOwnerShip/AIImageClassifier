import tensorflow as tf 
import numpy as np 
import pickle 
import matplotlib.pyplot as plt   


# load and proccess image 
# must change the path manually...
image = tf.keras.preprocessing.image.load_img('./ImgTestHuman/imageTest878.jpg', color_mode='grayscale')
image_array = tf.keras.preprocessing.image.img_to_array(image)  
image_array = np.expand_dims(image_array, axis=0)  # Agregar dimensión adicional 
 

# load trained model
with open('./trained_model-10-16.pkl', 'rb') as file:
    trainedModel = pickle.load(file)


# make prediction with image
prediction = trainedModel.predict(image_array) 

# the output is float from 0. to 1. 
# 0 for dog and 1 for food. 
# round the output to determine the result
if prediction >= 0.5:
    prediction_label = 1
    tag = 'food'
else:
    prediction_label = 0
    tag = 'dog'


# print prediction
print('prediction ', prediction)
print('prediction_label ', prediction_label)


#print image for human verification !
vvv = tf.keras.preprocessing.image.array_to_img(image_array[0])
plt.imshow(vvv, cmap='gray')
plt.title(str(prediction) + ' => ' + str(prediction_label) + ' :: ' + tag)
plt.show()
  
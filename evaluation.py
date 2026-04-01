
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries
from keras.models import load_model # This is used to load your saved model
from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import to_categorical # changed this because of compilation error

# Step 2: Load and return training and test datasets
def load_dataset():
	# 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# 2b. reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
	
	# 2c. normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
	X_test = X_test / 255
	
	# 2d. Convert y_train and y_test to categorical classes - Hint: y_train = np_utils.to_categorical(y_train)
	y_test = to_categorical(y_test)
	
	# 2e. return your X_train, X_test, y_train, y_test
	return X_test, y_test


# Step 3: Load your saved model 
model = load_model('digitRecognizer.h5')


# Step 4: Evaluate your model via your_model_name.evaluate(X_test, y_test, verbose = 0) function
X_test, y_test = load_dataset()
score = model.evaluate(X_test, y_test, verbose = 0)
print("Model accuracy: %.2f%%" % (score[1] * 100))


# Code below to make a prediction for a new image.


# Step 5: This section below is optional and can be copied from your digitRecognizer.py file from Step 8 onwards - load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

 
# Step 6: load and normalize new image
def load_new_image(path):
	# 6a. load new image
	newImage = load_img(path, color_mode = 'grayscale', target_size = (28, 28))
	# 6b. Convert image to array
	newImage = img_to_array(newImage)
	# 6c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
	newImage = newImage.reshape(1, 28, 28, 1)
	# 6d. normalize image data - Hint: newImage = newImage / 255
	newImage = newImage / 255.0
	# 6e. return newImage
	return newImage
 

# Step 7: load a new image and predict its class
def test_model_performance():
	# 7a. Call the above load image function
	img = load_new_image('digit1.png')
	# 7b. load your CNN model (digitRecognizer.h5 file)
	model = load_model('digitRecognizer.h5')	
	# 7c. predict the class - Hint: imageClass = your_model_name.predict_classes(img)
	prediction = model.predict(img)
	imageClass = prediction.argmax(axis=-1)
	# 7d. Print prediction result
	print("Predicted digit:", imageClass[0])


# Step 8: Test model performance here by calling the above test_model_performance function
test_model_performance()
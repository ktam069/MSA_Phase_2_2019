from phase2_ai import *

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.mobilenet import MobileNet

from keras.models import Model
from keras.layers import GlobalAveragePooling2D

from PIL import Image
from io import BytesIO

USE_DEFAULT_RESNET = False

def create_modified_resnet():
	# Load pre-trained ResNet without the last layer
	resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
	# resnet_model = MobileNet(include_top=False, input_shape=(32, 32, 3))
	
	latest_layer = resnet_model.output
	
	# latest_layer = GlobalAveragePooling2D()(latest_layer)
	# latest_layer = Flatten()(latest_layer)
	
	# latest_layer = Dense(64, activation='relu')(latest_layer)
	
	# Add output (dense layer)
	output_layer = Dense(10, activation='softmax')(latest_layer)    # 10 output classes - for CIFAR-10
	
	model = Model(inputs=resnet_model.input, outputs=output_layer)
	
	# Do not train the layers in the original model
	for layer in resnet_model.layers:
		layer.trainable = False
	
	model.summary()
	
	# (Sparse for integer encoded y_train, as opposed to one-hot encoded.)
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model

def create_default_resnet():
	# Load pre-trained ResNet without the last layer
	resnet_model = ResNet50(include_top=True, weights='imagenet')
	model = resnet_model
	
	# Do not train the layers in the original model
	for layer in resnet_model.layers:
		layer.trainable = False
	
	model.summary()
	
	# (Sparse for integer encoded y_train, as opposed to one-hot encoded.)
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
	return model

def run_resnet(x_train, y_train, x_test, y_test):
	if not USE_DEFAULT_RESNET:
		# Modified ResNet50 model, adapted to work with the CIFAR-10 data
		model = create_modified_resnet()
		
		# TODO: hyperparameter tuning (simply using the validation result currently)
		training = model.fit(x_train, y_train, batch_size=1000, epochs=20, validation_split=0.2)
		
		# Save the model so it can be loaded if desired (rather than having to re-train)
		save_trained_model(model, "resnet_cifar10")
		
		# Plot curves of training loss and validation loss
		plot_training_result(training)
	else:
		model = create_default_resnet()
	
	# Make predictions on test set
	y_pred = model.predict(x_test, verbose=1)
	
	# Get the class (i.e. index) with the highest probability
	y_pred = np.argmax(y_pred, axis=1)
	
	display_confusion_matrix(y_test, y_pred)
	
	eval_model(model, x_test, y_test)


def rescale_img_dataset(dataset, new_size=(224, 224)):
	new_shape = (dataset.shape[0],) + new_size + (dataset.shape[-1],)
	new_dataset = np.zeros(new_shape)
	
	for i in range(dataset.shape[0]):
		print("\tReshaping image data %d..."%i, end='\r', flush=True)
		# display_images(dataset[i:i+1,:,:,:])
		new_dataset[i,:,:,:] = rescale_img(dataset[i,:,:,:], new_size)
		# display_images(new_dataset[i:i+1,:,:,:])
	
	return new_dataset

def rescale_img(img_npy, new_size=(224, 224)):
	img_npy = Image.fromarray(img_npy.astype('uint8'), "RGB")
	img = img_npy.resize(new_size, Image.ANTIALIAS)
	
	return np.array(img)

def display_images(data):
	for i in range(data.shape[0]):
		imgplot = plt.imshow(data[i,:,:,:])
		plt.axis("off")
		plt.show()

def main():
	print("- Program running -")
	
	# Set up folder structure as needed
	setup()
	
	# Load dataset
	# x_train, y_train, x_test, y_test, label_names = load_data_batch()
	x_train, y_train, x_test, y_test, label_names = load_data()
	
	# Reshape the dataset into a shape that tensorflow expects: (samples, rows, cols, channels)
	# Note that the original data is stored as a flattened array in row-major order
	
	print(x_train.shape, '->')
	x_train = x_train.reshape(-1, 3, 32, 32)
	print(x_train.shape, '->')
	x_train = np.moveaxis(x_train, 1, -1)
	print(x_train.shape)
	
	x_test = x_test.reshape(-1, 3, 32, 32)
	x_test = np.moveaxis(x_test, 1, -1)
	
	print("\n=======\nNew shape:")
	print(x_train.shape)
	
	if USE_DEFAULT_RESNET:
		num_inputs = 10
		x_train = x_train[:num_inputs]
		y_train = y_train[:num_inputs]
		x_test = x_test[:num_inputs]
		y_test = y_test[:num_inputs]
		
		# display_images(x_test)
	
		x_train = rescale_img_dataset(x_train)
		x_test = rescale_img_dataset(x_test)
	
	# Preprocess the inputs
	x_train = preprocess_input(x_train)
	x_test = preprocess_input(x_test)
	
	run_resnet(x_train, y_train, x_test, y_test)
	
	print("- Program ended -")


if __name__ == '__main__':
	main()
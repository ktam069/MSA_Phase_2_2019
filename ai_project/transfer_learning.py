from phase2_ai import *

from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

from keras.models import Model
from keras.layers import GlobalAveragePooling2D

def create_modified_resnet():
	# Load pre-trained ResNet without the last layer
	resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
	# resnet_model = MobileNet(include_top=False, input_shape=(32, 32, 3))
	
	latest_layer = resnet_model.output
	
	# Flatten
	latest_layer = GlobalAveragePooling2D()(latest_layer)
	
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

def run_resnet(x_train, y_train, x_test, y_test):
	if USE_LOADED_MODEL:
		model = load_newest_model()
	else:
		# Modified ResNet50 model, adapted to work with the CIFAR-10 data
		model = create_modified_resnet()

		# TODO: hyperparameter tuning (simply using the validation result currently)
		training = model.fit(x_train, y_train, batch_size=1000, epochs=10, validation_split=0.2)
		
		# Save the model so it can be loaded if desired (rather than having to re-train)
		save_trained_model(model, "resnet_cifar10")
		
		# Plot curves of training loss and validation loss
		plot_training_result(training)
	
	# Make predictions on test set
	y_pred = model.predict(x_test, verbose=1)
	
	# Get the class (i.e. index) with the highest probability
	y_pred = np.argmax(y_pred, axis=1)
	
	display_confusion_matrix(y_test, y_pred)
	
	eval_model(model, x_test, y_test)

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
	
	# Normalise data into the range [0,1]
	x_train = x_train / np.max(x_train)
	x_test = x_test / np.max(x_test)
	
	run_resnet(x_train, y_train, x_test, y_test)
	
	print("- Program ended -")


if __name__ == '__main__':
	main()
from phase2_ai import *

import requests
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg

# Set image url here (for classifying)
url = "http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png"

def run_classifier(x_test, saved_model_name=None):
	USE_LOADED_MODEL = True

	if saved_model_name is not None:
		model = load_specific_model(saved_model_name)
		
		# Ignore invalid saved model name
		if model is None:
			print("Failed to load model, continuing with alternative models...\n")
			# run_CNN(x_train, y_train, x_test, y_test)
	elif USE_LOADED_MODEL:
		model = load_newest_model()
	elif USE_CHECKPOINT:
		model = load_newest_checkpoint()
	else:
		# Convolution NN model
		model = create_compiled_model()
		
		# Create checkpoints whenever validation accuracy has increased
		t = datetime.now().strftime("%d_%m_%H%M%S")
		filepath = checkpoint_save_dir + "cnn-%s-{epoch:02d}-{val_acc:.2f}.hdf5" %t
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
		callback_list = [checkpoint]
		
		# Hyperparameter tuning done simply by using the validation result
		training = model.fit(x_train, y_train, batch_size=200, epochs=15, callbacks=callback_list, validation_split=0.2)
		
		# Save the model so it can be loaded if desired (rather than having to re-train)
		save_trained_model(model)
		
		# Plot curves of training loss and validation loss
		plot_training_result(training)
	
	# Make predictions on test set
	print("\nPredicting on test set...")
	y_pred = model.predict(x_test, verbose=1)
	
	# Get the class (i.e. index) with the highest probability
	y_pred = np.argmax(y_pred, axis=1)

	return y_pred


response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.thumbnail((32, 32), Image.ANTIALIAS)

data = np.array(img)
data = data.reshape(1, 32, 32, 3)

output = run_classifier(data)[0]

label_names = unpickle(CIFAR_10_dir + "batches.meta")[b"label_names"]
label_names = [n.decode('utf-8') for n in label_names]

print("\n" + "="*60 + "\n")
print("Prediction on image:\n\t", url, "\n")
print("\t", label_names[output], "(class "+str(output)+")")
print()

imgplot = plt.imshow(img)
plt.show()

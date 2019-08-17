from phase2_ai import *

import requests
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
import sys

ASK_FOR_INPUT = True
saved_model_name = "saved_model_4conv_2nn_paper.h5"

'''Set image url here (for classifying)'''
# url = "http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png"
url = "https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"


def run_classifier(model, x_test):
	# Make predictions on test set
	print("\nPredicting on test set...")
	y_pred = model.predict(x_test, verbose=1)
	
	# Get the class (i.e. index) with the highest probability
	y_pred = np.argmax(y_pred, axis=1)

	return y_pred

def classify_img(model, url=url):
	response = requests.get(url)
	img_original = Image.open(BytesIO(response.content))
	img = img_original.resize((32, 32), Image.ANTIALIAS)

	data = np.array(img)
	data = data.reshape(1, 32, 32, 3)

	output = run_classifier(model, data)[0]

	label_names = unpickle(CIFAR_10_dir + "batches.meta")[b"label_names"]
	label_names = [n.decode('utf-8') for n in label_names]

	print("\n" + "="*60 + "\n")
	print("Prediction on image:\n\t", url, "\n")
	print("\t", label_names[output], "(class "+str(output)+")")
	print()

	imgplot = plt.imshow(img_original)
	plt.title("%s"%label_names[output])
	plt.axis("off")
	plt.show()

	
model = load_specific_model(saved_model_name)

# Ignore invalid saved model name
if model is None:
	print("Failed to load model,", saved_model_name, "\n")
	sys.exit(0)

if ASK_FOR_INPUT:
	while True:
		try:
			url = input("\nPlease input URL to image: ")
			
			classify_img(model, url)
		except KeyboardInterrupt:
			break
		except Exception as e:
			print("Unable to classify image from", url)
			print(e)
else:
	classify_img(model)

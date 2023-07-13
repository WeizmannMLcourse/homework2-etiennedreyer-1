import os

def test():

	if not os.path.exists('dog_breeds'):
		os.system('wget https://www.dropbox.com/s/nseb35op6ak7zvm/dog-breeds.zip')
		os.system('unzip dog-breeds.zip')

	from evaluate_dogs import evaluate_on_dataset

	loss = evaluate_on_dataset()
	threshold = 0.015

	print("Your loss: {}".format(loss))

	assert loss < threshold, "Loss needs to be less than {}".format(threshold)

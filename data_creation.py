# toy example: email addresses

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

extra = ['.','@']
total = alphabet + extra
mapping = {}
for i in range(len(total)):
	mapping[i] = total[i]
	mapping[total[i]] = i
import random
import numpy as np

def generate_n_true_samples(n,size):
	samples = []
	assert size>5
	for i in xrange(n):
		# using convention that pre-@ and post-@ are same size, we need to pick  (size - 5 ) letters
		letters = random.sample(alphabet, size-5)
		start = ''.join(letters[:len(letters)/2])
		end = ''.join(letters[len(letters)/2:])
		samples.append(start + '@' + end + '.com')

	return samples


def generate_n_false_samples(n,size):
	samples = []
	assert size>5
	for i in xrange(n):
		letters = random.sample(total, size)
		samples.append(''.join(letters))

	return samples

def string_to_array(string):
	X = np.zeros((len(string), len(total)))
	for i in range(len(X)):
		X[i,mapping[string[i]]] = 1
	return X


def generate_train_set(minibatch_size, num_minibatches):
	train_set = []
	base_size = 10
	for i in xrange(num_minibatches):
		curr_size = base_size + i
		true = [string_to_array(x) for x in generate_n_true_samples(minibatch_size/2, curr_size)]
		false = [string_to_array(x) for x in generate_n_false_samples(minibatch_size/2,curr_size)]
		concat = true+false
		minibatch = np.array(concat)
		y_vec = np.zeros(minibatch_size)
		for i in range(minibatch_size/2):
			y_vec[i] = 1
		train_set.append((minibatch,y_vec))
	return train_set



if __name__ == '__main__':
	true = generate_n_true_samples(2,10)
	false = generate_n_false_samples(2,10)
	ts = generate_train_set(10, 10)
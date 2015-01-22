from models import *
from data_creation import *
import time
import theano
import pickle

print "Generating train set..."
# dfa = theanoDFA(15, len(total))
if theano.config.floatX == 'float32':
	ts = generate_train_set_gpu(10000,10)
else:
	ts = generate_train_set(10000,10)
print "Initializing dfa 1..."
dfa= theanoDFA(15,len(total),init=2)
print "Initializing dfa 2..."
dfa_2 = theanoDFA(6,len(total),init=2)
print "Iterating through epochs..."
for epoch in xrange(1000):
	tic = time.time()
	for X,y in ts:
		O,F,transition_tensor,cost = dfa.trainmb(X,y)
		O2,F2,transition_tensor2,cost2 = dfa_2.trainmb(X,y)
	toc = time.time()
	print "Epoch %s took %s seconds." % (str(epoch), str(toc - tic))
	if (epoch+1) % 100 == 0:
		pickle.dump(dfa, open('dfa1_epoch_%s.pkl' % epoch,'w'))
		pickle.dump(dfa_2, open('dfa2_epoch_%s.pkl' % epoch, 'w'))

pickle.dump(dfa, open('dfa1_epoch_final.pkl','w'))
pickle.dump(dfa_2, open('dfa2_epoch_final.pkl','w'))


	
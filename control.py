from models import *
from data_creation import *
import time
import theano
import pickle


# dfa = theanoDFA(15, len(total))
if theano.config.floatX == 'float32':
	ts = generate_train_set_gpu(10000,10)
else:
	ts = generate_train_set(10000,10)
dfa= theanoDFA(15,len(total),init=2)
dfa_2 = theanoDFA(6,len(total),init=2 )
for epoch in xrange(1000):
	tic = time.time()
	for X,y in ts:
		O,F,transition_tensor,cost = dfa.trainmb(X,y)
		O2,F2,transition_tensor2,cost2 = dfa_2.trainmb(X,y)
	toc = time.time()
	print "Epoch %s took %s seconds." % (str(epoch), str(toc - tic))
	if (epoch+1) % 100 == 0:
		pickle.dump(dfa, open('epoch_%s.pkl' % epoch,'w'))

pickle.dump(dfa, open('epoch_final.pkl','w'))


	
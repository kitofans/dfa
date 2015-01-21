from models import *
from data_creation import *
import time
# dfa = theanoDFA(15, len(total))
ts = generate_train_set(10000,10)
dfa= theanoDFA(15,len(total),init=2)
for epoch in xrange(200):
	tic = time.time()
	for X,y in ts:
		O,F,transition_tensor,cost = dfa.trainmb(X,y)
		# O2,F2,transition_tensor2,cost2 = dfa_2.trainmb(X,y)
	toc = time.time()
	print "Epoch %s took %s seconds." % (str(epoch), str(toc - tic))
	
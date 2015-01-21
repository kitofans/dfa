'''
models for the dfa learner
ankitk@stanford.edu
'''
import theano
import theano.tensor as T
import numpy as np



def initT(n,m):
    # initialized an nxm weight matrix
    # copied from karpathy
    magic_number = 0.1
    return (np.random.rand(n,m,m) * 2 - 1) * magic_number

def initT2(n,m):
    # initialized an nxm weight matrix
    # copied from karpathy
    magic_number = .1
    return (np.random.rand(n,m,m) * 2) * magic_number

def initV(n):
    return np.random.rand(n) - .5

class dfa(object):

    def __init__(self, num_states, alphabet_size):
        # T is alphabet_size x num_states x num_states
        self.T = initT(alphabet_size, num_states)
        # 1 is the start state
        self.start_state = np.zeros(num_states)
        self.start_state[0] = 1
    def forward(X):
        # X is seq_size X alphabet_size one-hot representation
        TT = np.tensordot(X, self.T, 1)
        # TT is seq_size X num_states x num_states




class theanoDFA(object):

    def __init__(self, num_states, alphabet_size, clip_threshold=2, momentum=0, lr=0.01,init=1, regularization=1):
        # learnable params
        if init==1:
            self.WT = theano.shared(initT(alphabet_size, num_states))
        else:
            self.WT = theano.shared(initT2(alphabet_size,num_states))
        # state_definition should be continuous from 0 to 1 based on reject to accept
        self.state_definition = theano.shared(initV(num_states))


        self.L2_sqr = 0
        self.L2_sqr += (self.WT ** 2).sum()
        self.L2_sqr += (self.state_definition**2).sum()
        
        # scale it
        self.L2_sqr *= regularization


        self.params = [self.WT, self.state_definition]
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
        

        # fixed params
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.clip_threshold = clip_threshold
        self.momentum=momentum
        self.lr = lr

        # build funcs
        self.functions()
        self.functions_minibatch()


    def functions_minibatch(self):
        # instantiate symbolic variables
        X = T.tensor3()
        # y is accept or reject; 0 or 1.
        y = T.vector()

        # ========== [define forward pass] ==========

        # X is minibatch_size x seq_size x alphabet_size
        # self.WT is alphabet_size x num_states x num_states
        # hence transition_tensor is minibatch_size x seq_size x num_states x num_states
        transition_tensor = T.tensordot(X, self.WT, 1)
        # overall scan func:
        def one_sample(ex):
            # we need to define a recurrent function in here
            # state0 is starting state
            state0 = np.zeros(self.num_states)
            state0[0] = 1

            def recurrence(T_t, s_tm1):
                '''
                T_t is the transition matrix at this timestep; s_tm1 is the state from the timestep before
                '''

                '''todo: T_t should be column-wise probability distribution'''
                pdist = T_t / T.sum(T_t,axis=1,keepdims=True)
                s_t = T.dot(s_tm1,pdist) # TODO: prob dist
                return s_t

            S, _ = theano.scan(recurrence, sequences=ex, outputs_info=state0)
            Sfinal = S[-1]
            # now Sfinal = 1xnum_states
            return Sfinal
        F, _ = theano.scan(one_sample, sequences=transition_tensor)
        # now F is minibatch_size x num_states
        # F_tensor is minibatch_size x num_states 

        
        O = T.nnet.sigmoid(theano.dot(F, self.state_definition))
        # so O is minibatch_size x 1
        
        ''' todo: cost; tothink: use sigmoid in order to not require nonnegative components'''
        cost = T.nnet.binary_crossentropy(O, y).mean() + self.L2_sqr

        gparams = []
        # nonfiniteg = []
        for param in self.params:
            pre_clip_gparam = T.grad(cost, param)
            # calculate norm
            gparam_norm = T.sqrt(T.sum(pre_clip_gparam**2))
            # switch statement for the clipping
            switchgrad = T.switch(T.ge(gparam_norm, self.clip_threshold), pre_clip_gparam * (self.clip_threshold/gparam_norm), pre_clip_gparam)
            # make sure we have reasonable gradients
            nonfinite = T.or_(T.isnan(switchgrad), T.isinf(switchgrad))
            post_clip_gparam = T.switch(nonfinite, .01 * param, switchgrad)
            gparams.append(pre_clip_gparam)
            # nonfiniteg.append(switchgrad)



        updates = []
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = self.momentum * weight_update - self.lr * gparam
            updates.append((weight_update, upd))
            updates.append((param, param + upd))
        gparams.append(T.grad(cost,O))

        self.trainmb = theano.function([X,y], [O,F,transition_tensor,cost], updates=updates)
        self.predictmb =theano.function([X,y],[O,F,cost])



    def functions(self):
        # instantiate symbolic variables
        X = T.matrix()
        # y is accept or reject; 0 or 1.
        y = T.scalar()

        # ========== [define forward pass] ==========

        # X is seq_size x alphabet_size
        # self.WT is alphabet_size x num_states x num_states
        # hence transition_tensor is seq_size x num_states x num_states
        transition_tensor = T.tensordot(X, self.WT, 1)

        
        # we need to define a recurrent function in here
        # state0 is starting state
        state0 = np.zeros(self.num_states)
        state0[0] = 1

        def recurrence(T_t, s_tm1):
            '''
            T_t is the transition matrix at this timestep; s_tm1 is the state from the timestep before
            '''
            pdist = T_t / T.sum(T_t,axis=1,keepdims=True)
            s_t = T.dot(s_tm1,pdist)
            return s_t

        S, _ = theano.scan(recurrence, sequences=transition_tensor, outputs_info=state0)
        Sfinal = S[-1]
        # now Sfinal = 1xnum_states
        O = T.nnet.sigmoid(theano.dot(Sfinal, self.state_definition))
        # final is scalar 0-1
        ''' todo: cost; tothink: use sigmoid in order to not require nonnegative components'''
        cost = T.nnet.binary_crossentropy(O, y)

        gparams = []
        # nonfiniteg = []
        for param in self.params:
            pre_clip_gparam = T.grad(cost, param)
            # calculate norm
            gparam_norm = T.sqrt(T.sum(pre_clip_gparam**2))
            # switch statement for the clipping
            switchgrad = T.switch(T.ge(gparam_norm, self.clip_threshold), pre_clip_gparam * (self.clip_threshold/gparam_norm), pre_clip_gparam)
            # make sure we have reasonable gradients
            nonfinite = T.or_(T.isnan(switchgrad), T.isinf(switchgrad))
            post_clip_gparam = T.switch(nonfinite, .01 * param, switchgrad)
            gparams.append(post_clip_gparam)
            # nonfiniteg.append(switchgrad)



        updates = []
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = self.momentum * weight_update - self.lr * gparam
            updates.append((weight_update, upd))
            updates.append((param, param + upd))
        gparams.append(T.grad(cost,O))

        self.train = theano.function([X,y], [O,cost], updates=updates)
        self.predict =theano.function([X,y],[O,S,Sfinal,transition_tensor,cost])



if __name__ == '__main__':
    dfa = theanoDFA(10,5)
    testX = np.zeros((3,5))
    testX[0,1]=1
    testX[1,2]=1
    testX[2,0] = 1

    testX2 = np.zeros((3,5))
    testX2[0,3] = 1
    testX2[1,4] = 1
    testX2[2,2] = 1
    testy = 1

    paddedtestX = np.zeros((10,5))
    paddedtestX[0,1] = 1
    paddedtestX[1,2] = 1
    paddedtestX[2,0] = 1
    for i in range(3,10):
        paddedtestX[i,4] = 1
    final,cost = dfa.train(testX, testy)

    final1 = dfa.predict(testX,testy)
    final2 = dfa.predict(paddedtestX,testy)









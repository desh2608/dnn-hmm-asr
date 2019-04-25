#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import numpy as np

# neural network related
from sklearn.neural_network import MLPClassifier

def elog(x):
    res = np.log(x, where=(x!=0))
    res[np.where(x==0)] = -(10.0**8)
    return (res)

def get_data_dict(data):
    data_dict = {}
    for line in data:
        if "[" in line:
            key = line.split()[0]
            mat = []
        elif "]" in line:
            line = line.split(']')[0]
            mat.append([float(x) for x in line.split()])
            data_dict[key]=np.array(mat)
        else:
            mat.append([float(x) for x in line.split()])
    return data_dict


def logSumExp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return (x_max + np.log(sumexp))

def exp_normalize(x, axis=None, keepdims=False):
    b = x.max(axis=axis, keepdims=keepdims)
    y = np.exp(x - b)
    return y / y.sum(axis=axis, keepdims=keepdims)

def compute_ll(data, mu, r):
    # Compute log-likelihood of a single n-dimensional data point, given a single
    # mean and variance
    ll = (- 0.5*elog(r) - np.divide(
            np.square(data - mu), 2*r) -0.5*np.log(2*np.pi)).sum()
    return ll

def forward(pi, a, o, mu, r):
    """
    Computes forward log-probabilities of all states
    at all time steps.
    Inputs:
    pi: initial probability over states
    a: transition matrix
    o: observed n-dimensional data sequence
    mu: means of Gaussians for each state
    r: variances of Gaussians for each state
    """
    T = o.shape[0]
    J = mu.shape[0]

    log_alpha = np.zeros((T,J))
    log_alpha[0] = elog(pi)

    log_alpha[0] += np.array([compute_ll(o[0],mu[j],r[j])
        for j in range(J)])

    for t in range(1,T):
        for j in range(J):
            log_alpha[t,j] = compute_ll(o[t],mu[j],r[j]) + logSumExp(elog(a[:,j].T) + log_alpha[t-1])

    return log_alpha

def backward(a, o, mu, r):
    """
    Computes backward log-probabilities of all states
    at all time steps.
    Inputs:
    a: transition matrix
    o: observed n-dimensional data
    mu: means of Gaussians for each state
    r: variances of Gaussians for each state
    """
    T = o.shape[0]
    J = mu.shape[0]
    log_beta = np.zeros((T,J))

    log_a = elog(a)

    for t in reversed(range(T-1)):
        for i in range(J):
            x = []
            for j in range(J):
                x.append(compute_ll(o[t+1], mu[j], r[j]) + log_beta[t+1,j] + log_a[i,j])

            log_beta[t,i] = logSumExp(np.array(x))

    return log_beta

def getExpandedData(data):
    T = data.shape[0]
    data_0 = data[0]
    data_T = data[T-1]
    for i in range(3):
        data = np.insert(data, 0, data_0, axis=0)
        data = np.insert(data, -1, data_T, axis=0)

    data_expanded = []
    for t in range(3, T+3):
        frame_expanded = np.concatenate((data[t-3], data[t-2], data[t-1], data[t],
            data[t+1], data[t+2], data[t+3]))
        data_expanded.append(frame_expanded)
    return (np.array(data_expanded))

class SingleGauss():
    def __init__(self):
        # Basic class variable initialized, feel free to add more
        self.dim = None
        self.mu = None
        self.r = None

    def train(self, data):
        # Function for training single modal Gaussian
        T, self.dim = data.shape

        self.mu = np.mean(data, axis=0)
        self.r = np.mean(np.square(np.subtract(data, self.mu)), axis=0)
        return 

    def loglike(self, data_mat):
        # Function for calculating log likelihood of single modal Gaussian
        lls = [compute_ll(frame, self.mu, self.r) for frame in data_mat.tolist()]
        ll = np.sum(np.array(lls))
        return ll


class HMM():

    def __init__(self, sg_model, nstate):
        # Basic class variable initialized, feel free to add more
        self.pi = np.zeros(nstate)
        self.pi[0] = 1
        self.nstate = nstate

        self.mu = np.tile(sg_model.mu, (nstate,1))
        self.r = np.tile(sg_model.r, (nstate,1))


    def initStates(self, data):
        self.states = []
        for data_u in data:
            T = data_u.shape[0]
            state_seq = np.array([self.nstate*t/T for t in range(T)], dtype=int)
            self.states.append(state_seq)

    def getStateSeq(self, data):
        T = data.shape[0]
        J = self.nstate
        s_hat = np.zeros(T, dtype=int)
        
        log_delta = np.zeros((T,J))
        psi = np.zeros((T,J))
        
        log_delta[0] = elog(self.pi)
        for j in range(J):
            log_delta[0,j] += compute_ll(data[0], self.mu[j], self.r[j])

        log_A = elog(self.A)
        
        for t in range(1,T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t-1,i] + log_A[i,j] + compute_ll(data[t], self.mu[j], self.r[j])
                log_delta[t,j] = np.max(temp)
                psi[t,j] = np.argmax(log_delta[t-1]+log_A[:,j])


        s_hat[T-1] = np.argmax(log_delta[T-1])
        
        for t in reversed(range(T-1)):
            s_hat[t] = psi[t+1,s_hat[t+1]]

        return s_hat

        
    def viterbi(self, data):
        for u,data_u in enumerate(data):
            s_hat = self.getStateSeq(data_u)
            self.states[u] = s_hat


    def m_step(self, data):

        self.A = np.zeros((self.nstate,self.nstate))

        gamma_0 = np.zeros(self.nstate)
        gamma_1 = np.zeros((self.nstate, data[0].shape[1]))
        gamma_2 = np.zeros((self.nstate, data[0].shape[1]))
        
        for u, data_u in enumerate(data):
            T = data_u.shape[0]
            seq = self.states[u]
            gamma = np.zeros((T, self.nstate))

            for t,j in enumerate(seq[:-1]):
                self.A[j,seq[t+1]] += 1
                gamma[t,j] = 1

            gamma[T-1,self.nstate-1] = 1
            gamma_0 += np.sum(gamma, axis=0)

            for t in range(T):
                gamma_1[seq[t]] += data_u[t]
                gamma_2[seq[t]] += np.square(data_u[t])

        gamma_0 = np.expand_dims(gamma_0, axis=1)
        self.mu = gamma_1 / gamma_0
        self.r = (gamma_2 - np.multiply(gamma_0, self.mu**2))/ gamma_0

        for j in range(self.nstate):
            self.A[j] /= np.sum(self.A[j])



    def train(self, data, iter):
        # Function for training single modal Gaussian
        if (iter==0):
            self.initStates(data)
        self.m_step(data)
        self.viterbi(data)


    def loglike(self, data):
        # Function for calculating log likelihood of single modal Gaussian
        T = data.shape[0]
        log_alpha_t = forward(self.pi, self.A, data, self.mu, self.r)[T-1]
        ll = logSumExp(log_alpha_t)
            
        return ll


class HMMMLP():
    
    def __init__(self, mlp, hmm_model, S, uniq_state_dict):
        # Basic class variable initialized, feel free to add more
        self.mlp = mlp
        self.hmm = hmm_model
        self.log_prior = self.computeLogPrior(S)
        self.uniq_state_dict = uniq_state_dict


    def computeLogPrior(self, S):
        states, counts = np.unique(S, return_counts=True)
        p = np.zeros(len(states))
        for s,c in zip(states,counts):
            p[s] = c
        p /= np.sum(p)
        return elog(p)

    def mlp_predict(self, o):
        o_expanded = getExpandedData(o)
        return (self.mlp.predict_log_proba(o_expanded))


    def forward_dnn(self, pi, a, o, digit):

        T = o.shape[0]
        J = len(pi)

        log_alpha = np.zeros((T,J))
        log_alpha[0] = elog(pi)

        mlp_ll = self.mlp_predict(o)


        log_alpha[0] += np.array([mlp_ll[0][self.uniq_state_dict[(digit,j)]] - self.log_prior[self.uniq_state_dict[(digit,j)]]
            for j in range(J)])

        for t in range(1,T):
            for j in range(J):
                mlp_ll_t = mlp_ll[t][self.uniq_state_dict[(digit,j)]] - self.log_prior[self.uniq_state_dict[(digit,j)]]
                log_alpha[t,j] = mlp_ll_t + logSumExp(elog(a[:,j].T) + log_alpha[t-1])

        return log_alpha

    def loglike(self, data, digit):
        T = data.shape[0]
        log_alpha_t = self.forward_dnn(self.hmm.pi, self.hmm.A, data, digit)[T-1]
        ll = logSumExp(log_alpha_t)
            
        return ll



def sg_train(digits, train_data):
    model = {}
    for digit in digits:
        model[digit] = SingleGauss()

    for digit in digits:
        data = np.vstack([train_data[id] for id in train_data.keys() if digit in id.split('_')[1]])
        logging.info("process %d data for digit %s", len(data), digit)
        model[digit].train(data)

    return model


def hmm_train(digits, train_data, sg_model, nstate, niter):
    logging.info("hidden Markov model training, %d states, %d iterations", nstate, niter)

    hmm_model = {}
    data_dict = {}
    for digit in digits:
        hmm_model[digit] = HMM(sg_model[digit], nstate=nstate)
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        data_dict[digit] = data


    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        total_count = 0.0
        for digit in digits:
            data = data_dict[digit]
            logging.info("process %d data for digit %s", len(data), digit)

            hmm_model[digit].train(data, i)

            for data_u in data:
                total_log_like += hmm_model[digit].loglike(data_u)

        logging.info("log likelihood: %f", total_log_like)
        i += 1

    return hmm_model


def mlp_train(digits, train_data, hmm_model, uniq_state_dict, nepoch, lr, nunits=(256, 256), bsize=128):

    #TODO: Complete the function to train MLP and create HMMMLP object for each digit
    # Get unique output IDs for MLP, perform alignment to get labels and perform context expansion
    data_dict = {}

    # Get unique state sequences
    seq_dict = {}
    for digit in digits:
        uniq = lambda t: uniq_state_dict[(digit, t)]
        vfunc = np.vectorize(uniq)
        
        sequences = []
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        data_dict[digit] = data

        for data_u in data:
            seq = hmm_model[digit].getStateSeq(data_u)
            sequences.append(vfunc(seq))
        seq_dict[digit] = sequences

    # Perform context expansion and create large training matrix and labels
    O = []
    S = []
    for digit in digits:
        data = data_dict[digit]
        sequences = seq_dict[digit]
        for data_u, seq in zip(data, sequences):
            data_u_expanded = getExpandedData(data_u)
            O.append(data_u_expanded)
            S.append(seq)

    O = np.vstack(O)
    S = np.concatenate(S, axis=0)


    #TODO: A simple scikit-learn MLPClassifier call is given below, check other arguments and play with it
    #OPTIONAL: Try pytorch instead of scikit-learn MLPClassifier  
    mlp = MLPClassifier(hidden_layer_sizes=nunits, random_state=1, early_stopping=True, verbose=True,
        validation_fraction=0.1, shuffle=True, batch_size=128, tol=0.001, max_iter=nepoch)

    mlp.fit(O,S)

    mlp_model = {}
    for digit in digits:
        #TODO: variables to initialize HMMMLP are incomplete below, pass additional variables that are required
        mlp_model[digit] = HMMMLP(mlp, hmm_model[digit], S, uniq_state_dict)

    return mlp_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='test data')
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--nstate', type=int, default=5)
    parser.add_argument('--nepoch', type=int, default=5)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--mode', type=str, default='mlp',
                        choices=['hmm', 'mlp'],
                        help='Type of models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set seed
    np.random.seed(777)

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]
    states = range(args.nstate)
    uniq_state_dict = {}
    for i, digit in enumerate(digits):
        for state in states:
            uniq_state_dict[(digit, state)] = i*len(states) + state

    # read training data
    with open(args.train) as f:
        train_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        train_data = {key:train_data[key] for key in list(train_data.keys())[:200]}

    # read test data
    with open(args.test) as f:
        test_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        test_data = {key:test_data[key] for key in list(test_data.keys())[:200]}

    # Single Gaussian
    sg_model = sg_train(digits, train_data)

    if args.mode == 'hmm':
        model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
    elif args.mode == 'mlp':
        hmm_model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
	#TODO: Modify MLP training function call with appropriate arguments here
        model = mlp_train(digits, train_data, hmm_model, uniq_state_dict, nepoch=args.nepoch, lr=args.lr, 
            nunits=(256, 256), bsize=128)

    # test
    total_count = 0
    correct = 0
    for key in test_data.keys():
        lls = [] 
        for digit in digits:
            ll = model[digit].loglike(test_data[key], digit)
            lls.append(ll)
        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))

        logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
        if predict in key.split('_')[1]:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct)/total_count * 100)


# -*- coding: utf-8 -*-

from numpy import diag
import numpy.random as rand

import numpy as np
from util import rowStochastic, gnp, saveModelData
from ipyparallel import Client


def run_model(A):
    # The import needs to be done here
    # because the code runs in the engine
    import models
    return models.meetFriend_matrix(A, max_rounds, eps=1e-6)


if __name__ == '__main__':
    rand.seed(1233)
    N = 128
    max_rounds = 10000
    simid = 'mfm_gnp1'
    print('[*] N = {0}'.format(N))

    print('[*] Creating Networks...')
    p_list = np.arange(0.0, 1.0, 0.05)
    A_matrices = [gnp(N, p) for p in p_list]
    B = diag(rand.rand(N))
    networks = [rowStochastic(A+B) for A in A_matrices]
    print('[*] Running Simulations...')

    c = Client()
    v = c[:]
    print('[*] Calculating using {0} parallel engines...'.format(len(v)))
    v.block = True
    v.push(dict(max_rounds=max_rounds))
    result = v.map_sync(run_model, networks)

    print('[*] Saving Results...')
    saveModelData(simid, N=N, max_rounds=max_rounds, eps=1e-6,
                  rounds_run=max_rounds, A=A, result=np.array(result),
                  p_list=np.array(p_list))

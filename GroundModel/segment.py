import theano
import theano.tensor as tensor
from theano import function, config, shared, tensor
import numpy as np

import globals

#/////////////////////////////////////////////////////////
#//SEGMENT////////////////////////////////////////////////
#/////////////////////////////////////////////////////////

def do(j):

    getSpBins = theano.function([],globals.Sp.Bins)
    SpBins = getSpBins()
    SpBins = np.array(SpBins)

    getSpZ = theano.function([],globals.Sp.Z)
    SpZ = getSpZ()
    SpZ = np.array(SpZ)

    myBin = np.array(np.arange(SpBins.size),dtype=bool)

    for i in range(0,SpBins.size):

        if j == SpBins[i]:

            myBin[i] = True

        else:

            myBin[i] = False

    Z = shared(np.empty(0, dtype=config.floatX))

    Z = globals.Sp.Z[myBin]

    dim_Z = function([], Z.shape[0])
    dim_Z = dim_Z()

    if (dim_Z > 0):

        Hg = theano.tensor.mean(Z)

        getHg = theano.function([],Hg)
        Hg = getHg()

        for i in range(0,SpBins.size):

            if j == SpBins[i]:

                if abs(SpZ[i] - Hg) <= globals.Tg:

                    globals.Sp.Ground[i] = True

                else:

                    globals.Sp.Ground[i] = False



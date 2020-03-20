
import theano
import theano.tensor as tensor
from theano import function, config, shared, tensor
import numpy as np
import csv

class Seed:

    X = []
    Y = []
    Z = []
    OP = []
    idx = []

    def __call__(self):
      
        self.X = shared(np.empty(0, dtype=config.floatX))
        self.Y = shared(np.empty(0, dtype=config.floatX))
        self.Z = shared(np.empty(0, dtype=config.floatX))
        self.OP = shared(np.empty(0, dtype=config.floatX))
        self.idx = shared(np.empty(0, dtype=np.int16))
        self.Bins = shared(np.empty(0, dtype=np.int16))
        self.Ground = np.empty(0, dtype=bool)

    def extractSector(self,PGi):

        import globals

        globals.PGi()

        idx = theano.tensor.eq(globals.Sectors, PGi)

        globals.PGi.X = globals.X[idx]
        globals.PGi.Y = globals.Y[idx]
        globals.PGi.Z = globals.Z[idx]
        globals.PGi.OP = globals.op[idx]
        globals.PGi.idx = globals.idx[idx]
        globals.PGi.Bins = globals.Bins[idx]

    def extractSeed(self):

        import globals

        self.Z = theano.tensor.clip(globals.PGi.Z,globals.minZ,np.Inf)

        idx = theano.tensor.eq(self.Z, globals.PGi.Z).nonzero()[0]

        self.X = globals.PGi.X[idx]
        self.Y = globals.PGi.Y[idx]
        self.Z = globals.PGi.Z[idx]
        self.OP = globals.PGi.OP[idx]
        self.idx = globals.PGi.idx[idx]
        self.Bins = globals.PGi.Bins[idx]

    def extractTest(self):

        import globals

        getPGi = theano.function([],globals.PGi.idx)
        PGi = getPGi()
        PGi = np.array(PGi)

        getSp = theano.function([],globals.Sp.idx)
        Sp = getSp()
        Sp = np.array(Sp)

        #with open('sanityCheck/' + 'Test-PGi' + '.txt', mode='w') as pg_file:
        #            fieldsNames = ['PGi']
        #            pg_writer = csv.DictWriter(pg_file, fieldnames=fieldsNames, delimiter=',', lineterminator='\n')
        #            pg_writer.writeheader()
                
        #            for i in range(0,len(PGi)-1):
        #                pg_writer.writerow({'PGi':round(PGi[i],2)})

        #with open('sanityCheck/' + 'Test-Sp' + '.txt', mode='w') as pg_file:
        #            fieldsNames = ['Sp']
        #            pg_writer = csv.DictWriter(pg_file, fieldnames=fieldsNames, delimiter=',', lineterminator='\n')
        #            pg_writer.writeheader()
                
        #            for i in range(0,len(Sp)-1):
        #                pg_writer.writerow({'Sp':round(Sp[i],2)})

        exist = np.ones(PGi.size, dtype=bool)

        for i in range(0,PGi.size - 1):

            if (PGi[i] in Sp):

                exist[i] = False

            else:

                exist[i] = True

        self.X = globals.PGi.X[exist]
        self.Y = globals.PGi.Y[exist]
        self.Z = globals.PGi.Z[exist]
        self.OP = globals.PGi.OP[exist]
        self.idx = globals.PGi.idx[exist]
        self.Bins = globals.PGi.Bins[exist]

    def empty(self):

       self.X = None
       del self.X

       self.Y = None
       del self.Y

       self.Z = None
       del self.Z

       self.OP = None
       del self.OP

       self.idx = None
       del self.idx

       self.Bins = None
       del self.Bins

    def append(self,newSeed):

        self.X = theano.tensor.concatenate([self.X, newSeed.X], axis=0)
        self.Y = theano.tensor.concatenate([self.Y, newSeed.Y], axis=0)
        self.Z = theano.tensor.concatenate([self.Z, newSeed.Z], axis=0)
        self.OP = theano.tensor.concatenate([self.OP, newSeed.OP], axis=0)
        self.idx = theano.tensor.concatenate([self.idx, newSeed.idx], axis=0)
        self.Bins = theano.tensor.concatenate([self.Bins, newSeed.Bins], axis=0)

    def sanityCheck(self,printArrayInformation,saveArrayInformation,which):

        if printArrayInformation:

            # --- check X ---
            getX = theano.function([],self.X)
            x = getX()
            x = np.array(x)
            print(which)
            
            if printArrayInformation:
                print(x)

            # --- check Y ---
            getY = theano.function([],self.Y)
            y = getY()
            y = np.array(y)
            print(which)

            if printArrayInformation:
                print(y)

            # --- check Z ---
            getZ = theano.function([],self.Z)
            z = getZ()
            z = np.array(z)
            print(which)
            
            if printArrayInformation:
                print(z)

            # --- check OP ---
            getOP = theano.function([],self.OP)
            op = getOP()
            op = np.array(op)
            print(which)

            if printArrayInformation:
                print(op)

            # --- check IDX ---
            getIDX = theano.function([],self.idx)
            idx = getIDX()
            idx = np.array(idx)
            print(which)

            if printArrayInformation:
                print(idx)

            # --- check Bins ---
            getBINS = theano.function([],self.Bins)
            bins = getBINS()
            bins = np.array(bins)
            print(which)

            if printArrayInformation:
                print(bins)

            if saveArrayInformation:
                with open('sanityCheck/' + which + '.txt', mode='w') as pg_file:
                    fieldsNames = ['x', 'y', 'z', 'op', 'idx','bins']
                    pg_writer = csv.DictWriter(pg_file, fieldnames=fieldsNames, delimiter=',', lineterminator='\n')
                    pg_writer.writeheader()
                
                    for i in range(0,len(x)-1):
                        pg_writer.writerow({'x':round(x[i],2),'y':round(y[i],2),'z':round(z[i],2),'op':round(op[i],2),'idx':round(idx[i],2),'bins':round(bins[i],2)})


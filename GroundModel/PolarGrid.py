
#/////////////////////////////////////////////////////////////////////////
#//LOAD THE POLAR GRID 3D Data POINTS FROM BINARY FILE
#/////////////////////////////////////////////////////////////////////////

import os
import numpy as np
from struct import unpack
from theano import function, config, shared, tensor
from theano.ifelse import ifelse
import csv
import theano

def getSectorsAndBins():

    # -----------------------------------------------------------------------
    # LOAD BINARY DATA FROM 3D LIDAR VELODYNE LASER
    # -----------------------------------------------------------------------

    import globals

    NELEMENTS = 4
    NBYTES = 4

    fileSize = os.path.getsize(globals.fileRunning)
    fileElements = fileSize/NBYTES

    Xdata = np.zeros(int(fileElements/NELEMENTS), dtype=config.floatX)
    Ydata = np.zeros(int(fileElements/NELEMENTS), dtype=config.floatX)
    Zdata = np.zeros(int(fileElements/NELEMENTS), dtype=config.floatX)
    IDXdata = np.zeros(int(fileElements/NELEMENTS), dtype=config.floatX)

    fileData = np.fromfile(globals.fileRunning,float)

    iByte = 0
    with open(globals.fileRunning, 'rb') as file:
    
        chuck = file.read()

    for i in range(0,int(fileElements/NELEMENTS) - 1):

            [byte,] = unpack('<f', chuck[(i * NELEMENTS * NBYTES):(i * NELEMENTS * NBYTES) + NBYTES])
            Xdata[i] = byte
       
            [byte,] = unpack('<f', chuck[(i * NELEMENTS * NBYTES) + NBYTES:(i * NELEMENTS * NBYTES) + NBYTES + NBYTES])
            Ydata[i] = byte
       
            [byte,] = unpack('<f', chuck[(i * NELEMENTS * NBYTES) + (2 * NBYTES):(i * NELEMENTS * NBYTES) + (2 * NBYTES) + NBYTES])
            Zdata[i] = byte

            IDXdata[i] = i

    
    globals.X = shared(Xdata)
    globals.Y = shared(Ydata)
    globals.Z = shared(Zdata)

    globals.idx = shared(IDXdata)

    Xdata = None
    del Xdata
    Ydata = None
    del Ydata
    Zdata = None
    del Zdata
    IDXdata = None
    del IDXdata

    # -----------------------------------------------------------------------
    # GET SECTORS AND BINS
    # -----------------------------------------------------------------------

    globals.op = tensor.sqrt(globals.X * globals.X + globals.Y * globals.Y)

    globals.cos = globals.X / globals.op
    negate = tensor.switch(tensor.lt(globals.cos, 0), 1, 0)
    globals.cos = tensor.switch(tensor.lt(globals.cos, 0), -1*globals.cos, globals.cos)

    ret = float(-0.0187293)
    ret = ret * globals.cos
    ret = ret * globals.cos;
    ret = ret + float(0.0742610)
    ret = ret * globals.cos;
    ret = ret - float(0.2121144);
    ret = ret * globals.cos;
    ret = ret + float(1.5707288);
    ret = ret * tensor.sqrt(1.0-globals.cos);
    ret = ret - 2 * negate * ret;
    globals.acos = negate * globals.PI + ret;

    globals.sign = tensor.switch(tensor.lt(globals.Y, 0), -1, 1)
    globals.sign = tensor.switch(tensor.lt(globals.Y, 0) | tensor.lt(0, globals.Y), globals.sign, 0)

    globals.H = tensor.switch(tensor.lt(globals.Y*(-1), 0), 0, 1)
    globals.H = tensor.switch(tensor.lt(globals.Y*(-1), 0) | tensor.lt(globals.Y*(-1), 0), globals.H, 0.5)

    globals.exclude = tensor.switch(tensor.lt(globals.op, globals.R), 1, 0)

    globals.Sectors = tensor.ceil( ( ( ( globals.sign * globals.acos ) + (2 * globals.PI * globals.H) ) / globals.alpha ) )*globals.exclude;

    globals.Bins = tensor.ceil(globals.op / globals.Rstep)*globals.exclude;


def SanityCheckPolarGrid(printArrayInformation,saveArrayInformation):

     import globals

     if (printArrayInformation | saveArrayInformation):

         # --- check OP ---
         getX = function([],globals.X)
         x = getX()
         x = np.array(x)
         print('X')
         
         if printArrayInformation:
            print(x)

         # --- check OP ---
         getY = function([],globals.Y)
         y = getY()
         y = np.array(y)
         print('Y')

         if printArrayInformation:
            print(y)

         # --- check OP ---
         getZ = function([],globals.Z)
         z = getZ()
         z = np.array(z)
         print('Z')

         if printArrayInformation:
            print(z)

         # --- check OP ---
         getOP = function([],globals.op)
         op = getOP()
         op = np.array(op)
         print('OP')

         if printArrayInformation:
            print(op)

         # --- check acos ---
         getCos = function([],globals.cos)
         cos = getCos()
         cos = np.array(cos)
         print('cos')

         if printArrayInformation:
            print(cos)

         # --- check acos ---
         getAcos = function([],globals.acos)
         acos = getAcos()
         acos = np.array(acos)
         print('acos')

         if printArrayInformation:
            print(acos)

         # --- check sign ---
         getSign = function([],globals.sign)
         sign = getSign()
         sign = np.array(sign)
         print('sign')

         if printArrayInformation:
            print(sign)

         # --- check H ---
         getH = function([],globals.H)
         H = getH()
         H = np.array(H)
         print('H')

         if printArrayInformation:
            print(H)

         # --- check Exclude ---
         getExclude = function([],globals.exclude)
         exclude = getExclude()
         exclude = np.array(exclude)
         print('exclude')

         if printArrayInformation:
            print(exclude)

         # --- check Sectors ---
         getSectors = function([],globals.Sectors)
         sectors = getSectors()
         sectors = np.array(sectors)
         print('Sectors')

         if printArrayInformation:
            print(sectors)

         # --- check Bins ---
         getBins = function([],globals.Bins)
         bins = getBins()
         bins = np.array(bins)
         print('Bins')

         # --- check IDX ---
         getIDX = function([],globals.idx)
         idx = getIDX()
         idx = np.array(idx)
         print('idx')

         if printArrayInformation:
            print(idx)

         if saveArrayInformation:
            with open('sanityCheck/polarGrid.txt', mode='w') as pg_file:
                fieldsNames = ['X', 'Y', 'Z', 'OP', 'cos', 'acos', 'sign', 'H', 'exclude', 'Sectors', 'Bins', 'idx']
                pg_writer = csv.DictWriter(pg_file, fieldnames=fieldsNames, delimiter=',', lineterminator='\n')
                pg_writer.writeheader()
                
                for i in range(0,len(x)-1):
                    pg_writer.writerow({'X':round(x[i],2),'Y':round(y[i],2),'Z':round(z[i],2),'OP':round(op[i],2),'cos':round(cos[i],2),'acos':round(acos[i],2),'sign':round(sign[i],2),'H':round(H[i],2),'exclude':round(exclude[i],2),'Sectors':round(sectors[i],2),'Bins':round(bins[i],2),'idx':round(idx[i],2)})


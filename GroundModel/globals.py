
#/////////////////////////////////////////////////////////////////////////
#//DEFINE PARAMETERS
#/////////////////////////////////////////////////////////////////////////

import Seed
import numpy as np

#-------------------------------------------------------------------------
# MODEL PARAMETERS
#-------------------------------------------------------------------------

length_scale = np.dtype('float32').type(0.3)
sigma_f = np.dtype('float32').type(1.3298)
sigma_n = np.dtype('float32').type(0.1)
TData = np.dtype('float32').type(80/100)
TModel = np.dtype('float32').type(0.5/100)
Tg = np.dtype('float32').type(100/100)

PI = 3.1415
M = 360
R = 50
N = 160
Rstep = 0.3125
alpha = (2 * PI)/M

#-------------------------------------------------------------------------
# FILE DATA PARAMETERS
#-------------------------------------------------------------------------

fileRunning = "C:\\Users\\Dornas\\Dropbox\\__ D - BE-HAPPY\\y. HARD-QUALE\\_AREAS - RESEARCH\\KOD-DEMAND\\_CLIENTS\\_SR-INFO\\_DATA\\Velodyne\\laser\\000050.bin"

imageRunning = "C:\\Users\\Dornas\\Dropbox\\__ D - BE-HAPPY\\y. HARD-QUALE\\_AREAS - RESEARCH\\KOD-DEMAND\\_CLIENTS\\_SR-INFO\\_DATA\\Velodyne\\pictures\\Training\\000050.png"

folderRunning = "C:\\Users\\Dornas\\Hard Quale .com\\Kod-Demand - Documents\\_CLIENTS\\SEOULROBOTICS\\_DATA\\Velodyne\\Training"

folderCamCalibration = "C:\\Users\\Dornas\\Hard Quale .com\\Kod-Demand - Documents\\_CLIENTS\\SEOULROBOTICS\\_DATA\\Velodyne\\calib"

camTocam = folderCamCalibration + "\\" + "calib_cam_to_cam.txt"
veloTocam = folderCamCalibration + "\\" + "calib_velo_to_cam.txt"

#-------------------------------------------------------------------------
# VELODYNE DATA 
#-------------------------------------------------------------------------

X = []
Y = []
Z = []

idx = []

op = []

cos = []
acos = []

sign = []
H = []
exclude = []

Sectors = []
Bins = []

#-------------------------------------------------------------------------
# DEFINE MINIMUM VALUE OF Z FOR THRESHOLD
#-------------------------------------------------------------------------

minZ = -1.4

#-------------------------------------------------------------------------
# SECTORS DATA
#-------------------------------------------------------------------------

PGi = Seed.Seed()

Snew = Seed.Seed()

Sp = Seed.Seed()

Test = Seed.Seed()

#-------------------------------------------------------------------------
# SANITY CHECK PARAMETERS
#-------------------------------------------------------------------------

printArrayInformation = True
saveArrayInformation = True

filename = 'C:\Users\Dornas\Dropbox\__ D - BE-HAPPY\y. HARD-QUALE\_AREAS - RESEARCH\KOD-DEMAND\_CLIENTS\_SR-INFO\_DATA\Velodyne\laser\000000.bin';

fid = fopen(filename, 'r');
X = fread(fid, 2, 'float32');

OP = sqrt(X(1) * X(1) + X(2) * X(2));

cos = X(1) / OP;


aacos = acos(cos)

clear all

folder = 'C:\Users\Dornas\Dropbox\__ D - BE-HAPPY\y. HARD-QUALE\_AREAS - RESEARCH\KOD-DEMAND\_CLIENTS\_SR-CODE\TP-SeoulRobotics\GroundSegmentationModelPython\GroundModel';

filePG = 'polarGrid';

filePGi = 'PGi-1';

PG = csvread(strcat(folder,'\',filePG,'.txt'),1,0);

PGi = csvread(strcat(folder,'\',filePGi,'.txt'),1,0);

indices = csvread(strcat(folder,'\','indices','.txt'),1,0);

sector = 1;

PGi(:,5) = [];

PG(:,11) = [];
PG(:,5:9) = [];

idx_one = PG(:,5) == sector;

PG_one = PG(idx_one,1:4);

isequal(PG_one,PGi)


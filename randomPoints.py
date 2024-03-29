#!/usr/bin/python
from math import *
import numpy as np
import random
import sys
import os
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.core.operations import SymmOp

def distance(x1,y1,z1,x2,y2,z2,scale,a,b,c):
    i=((x1-x2)*a[0]+(y1-y2)*b[0]+(z1-z2)*c[0])
    j=((x1-x2)*a[1]+(y1-y2)*b[1]+(z1-z2)*c[1])
    k=((x1-x2)*a[2]+(y1-y2)*b[2]+(z1-z2)*c[2])
    dist=pow((pow(i,2)+pow(j,2)+pow(k,2)),0.5)*scale
    return (dist)

def checkDistance(x,y,z,pos,R,scale,a,b,c):
    trans = [0,1,-1]
    for i in pos:
        for tx in trans:
            for ty in trans:
                for tz in trans:
                    if distance(x+tx,y+ty,z+tz,float(i[0]),float(i[1]),float(i[2]),scale,a,b,c)<R:
                        return (1)
    return (0)

def checkSymmetry(x,y,z,Structures,symmops,a,b,c):
    tol = 0.0001#tolerance
    for site in Structures:
        #transforming fractional coord to cartesian coord
        i1=x*a[0]+y*b[0]+z*c[0]
        j1=x*a[1]+y*b[1]+z*c[1]
        k1=x*a[2]+y*b[2]+z*c[2]
        i2=site[0]*a[0]+site[1]*b[0]+site[2]*c[0]
        j2=site[0]*a[1]+site[1]*b[1]+site[2]*c[1]
        k2=site[0]*a[2]+site[1]*b[2]+site[2]*c[2]
        equ_points=[]#record equivalent points of i1,j1,k1
        for ops in symmops:
            equ_points.append(ops.operate([i1,j1,k1]))
        for equ_point in equ_points:
            if abs(i2-equ_point[0]) < tol and abs(j2-equ_point[1]) < tol and abs(k2-equ_point[2]) < tol:
                return 1
    return 0

#read symmetry information
'''
poscar = Poscar.from_file("CONTCAR")
structure = poscar.structure
Symmetry=SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
symmops=Symmetry.get_symmetry_operations(cartesian=True)
'''


nli = 0
for i in range(7737):
    if not os.path.isfile("select/%d"%(i)):
        continue
    poscar = Poscar.from_file("select/%d"%(i))
    structure = poscar.structure
    Symmetry=SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5)
    symmops=Symmetry.get_symmetry_operations(cartesian=True)
    f=open('poscar/POSCAR_%d'%(i),'r')
    #read lattice information
    title = f.readline().strip()
    scale = f.readline()
    scale = float(scale)
    a = f.readline().split()
    b = f.readline().split()
    c = f.readline().split()
    a = [float(j) for j in a]
    b = [float(j) for j in b]
    c = [float(j) for j in c]
    #read atom labels and number of atoms
    at_labels = f.readline().split()
    ats = f.readline().split()
    ats = [int(i) for i in ats]
    ntot = sum(ats)#total number of atoms
    if ntot > 100:
        continue
    f.readline()
    #read positions of atoms
    pos = []
    for j in range(ntot):
        d = f.readline().split()
        e = []
        for k in range(3):
            e.append(float(d[k]))
        pos.append(e)

    f.readline()

    #FFT grid
#    fft = [10000,10000,10000]#initialize FFT grids
    fft = [0,0,0]
    length_x=distance(0,0,0,1,0,0,scale,a,b,c); fft[0] = int(length_x/0.2)
    length_y=distance(0,0,0,0,1,0,scale,a,b,c); fft[1] = int(length_y/0.2)
    length_x=distance(0,0,0,0,0,1,scale,a,b,c); fft[2] = int(length_z/0.2)

    npoints=1
    for j in fft:
        npoints*=j#get total number of grids

    nStructures = 50#number of structures to generate

    Structures=[]
    rClose=2
    rFar = 5
    j=0
    rp = 0
    while j < nStructures:
        #randomly generate a point
        x=random.randint(1,fft[0])
        y=random.randint(1,fft[1])
        z=random.randint(1,fft[2])
        if checkDistance(x/fft[0],y/fft[1],z/fft[2],pos,rClose,scale,a,b,c):
            rp+=1
            if rp > 500:
                break
            print ('too close to the lattice')
            continue
        if not checkDistance(x/fft[0],y/fft[1],z/fft[2],pos,rFar,scale,a,b,c):
            rp+=1
            if rp > 500:
                break
            print ('too far to the lattice')
            continue
        if i !=0 and checkSymmetry(x/fft[0],y/fft[1],z/fft[2],Structures,symmops,a,b,c):
            print ('symmetrically equivalent')
            continue
        j += 1
        inserted_structure = structure.copy()
        inserted_structure.append("Li", [x/fft[0],y/fft[1],z/fft[2]])
        inserted_structure.to(filename = '4/%s_%s'%(str(i),str(j)), fmt='poscar')
        Structures.append([x/fft[0],y/fft[1],z/fft[2]])#record already selected points
        print ('%d, %d'%(i, j))
        nli += 1


print (nli)

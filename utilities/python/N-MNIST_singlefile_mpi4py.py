
print("here")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
procs = comm.Get_size()

from tqdm import tqdm
from read_functions import read_spike_np2, read_pot_np
import os
# import pandas as pd  # data handeling
import numpy as np   # numerical computing

import itertools  # combinatorics functions for multinomial code
import subprocess
# import pathlib 
import os
# from tqdm import tqdm
# from joblib import Parallel, delayed
# import multiprocessing
from time import time#, sleep

from random import shuffle
# print(rank,procs)
N=1000
gpM=0
gpS=0.2
gM=0
gS=0.1
cF=2
cB=1
cS=1
I=100
P=120
R=0
w=1

endname="_N({})_gp({},{})_g({},{})_c({},{},{})_IPR({},{},{})_w{}".format(N,gpM,gpS,gM,gS,cF,cB,cS,I,P,R,w)

# endname="_N({})_gp({},{})_g({},{})_c({},{},{})_IPR({},{},{})_w{}".format(N,gpM,gpS,gM,gS,cF,cB,cS,I,P,R,w)
# 
SPIKES=False
POTENTIAL=True
trfiles=None
tefiles=None
if rank==0:
    if SPIKES:
        trfiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TrainRCSp")) for f in fn]#[:1000]
        tefiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TestRCSp")) for f in fn]#[:1000]
    elif POTENTIAL:
        trfiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TrainRCPt")) for f in fn]#[:1000]
        tefiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TestRCPt")) for f in fn]#[:1000]
    else:
        print("No data selected")

    trfiles = [trfiles[i::procs] for i in range(procs)]
    tefiles = [tefiles[i::procs] for i in range(procs)]

    # num_cores = multiprocessing.cpu_count()

    # Nfiles=len(trfiles)
    # batchsize=Nfiles//procs

    X_train = [None]*N
    y_train = [None]*N

trfiles=comm.scatter(trfiles)
print(rank, trfiles[0])
tefiles=comm.scatter(tefiles)

start_time = time() 
def readtrfiles(fname,i,rank):
    # global X_train, y_train
    if SPIKES:
        this_x = read_spike_np2(fname,layer=1,N=N,binary=True)
    elif POTENTIAL:
        this_x = read_pot_np(fname,layer=1,p=28,N=N,binary=False)

    label = int(fname.split('/')[-2])
    this_y = (np.ones((this_x.shape[0],))*label).astype(np.int)
    end_time=time()
    my_time=end_time-start_time
    if i>0 and rank==0:
        ETA = (float(len(trfiles))/float(i) -1)*my_time
        ETAhr, ETAmn, ETAsc = int(ETA/3600), int(ETA/60)%60, int(ETA)%60
        MyThr, MyTmn, MyTsc = int(my_time/3600), int(my_time/60)%60, int(my_time)%60
        print("Finished {}  training files \n\nTime: {:02d}:{:02d}:{:02d}  ETA: {:02d}:{:02d}:{:02d} \n\n".format(i,MyThr,MyTmn,MyTsc,ETAhr,ETAmn,ETAsc))
        print(rank,procs)
    return i, this_x, this_y

par_return=[readtrfiles(f, i, rank) for (i,f) in enumerate(trfiles)]
# par_return=[readtrfiles(f, i) for (i,f) in enumerate(trfiles)]

par_return = comm.gather(par_return,root=0)
if rank==0:
    par_return = [fi for li in par_return for fi in li]
    for i,x,y in par_return:
        X_train[i]=x
        y_train[i]=y

    assert len(X_train)==len(y_train)
    d_train = list(zip(X_train,y_train))
    shuffle(d_train)
    X_train, y_train = zip(*d_train)
    X_train=np.concatenate(X_train,0)
    y_train=np.concatenate(y_train,0)

    if SPIKES:
        np.save("X_train_sp"+endname,X_train)
        np.save("y_train_sp"+endname,y_train)
    if POTENTIAL:
        np.save("X_train_pt"+endname,X_train)
        np.save("y_train_pt"+endname,y_train)

N=len(tefiles)

X_test = [None]*N
y_test = [None]*N

start_time = time() 
def readtefiles(fname,i,rank):
    if SPIKES:
        this_x = read_spike_np2(fname,layer=1,N=N,binary=True)
    elif POTENTIAL:
        this_x = read_pot_np(fname,layer=1,p=28,N=N,binary=False)
    label = int(fname.split('/')[-2])
    this_y=(np.ones((this_x.shape[0],))*label).astype(np.int)
    end_time=time()
    my_time=end_time-start_time
    if i>0 and rank==0:
        ETA = (float(N)/float(i) -1)*my_time
        ETAhr, ETAmn, ETAsc = int(ETA/3600), int(ETA/60)%60, int(ETA)%60
        MyThr, MyTmn, MyTsc = int(my_time/3600), int(my_time/60)%60, int(my_time)%60

        print("Finished {} testing files \n\nTime: {:02d}:{:02d}:{:02d}  ETA: {:02d}:{:02d}:{:02d} \n\n".format(i,MyThr,MyTmn,MyTsc,ETAhr,ETAmn,ETAsc))
    return i, this_x, this_y

par_return= [readtefiles(f,i, rank) for (i,f) in enumerate(tefiles)]
# par_return = comm.gather(par_return,root=0)

if rank==0:
    par_return = [fi for li in par_return for fi in li]
    for i,x,y in par_return:
       X_test[i]=x
       y_test[i]=y

    assert len(X_test)==len(y_test)
    d_test = list(zip(X_test,y_test))
    shuffle(d_test)
    X_test, y_test = zip(*d_test)
    X_test=np.concatenate(X_test,0)
    y_test=np.concatenate(y_test,0)


    if SPIKES:
        np.save("X_test_sp"+endname,X_test)
        np.save("y_test_sp"+endname,y_test)
    if POTENTIAL:
        np.save("X_test_pt"+endname,X_test)
        np.save("y_test_pt"+endname,y_test)

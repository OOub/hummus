import subprocess
import pathlib 
import os
# from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from time import time, sleep


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

#rcNetwork_N(100)_gp(0,0.2)_g(0,0.1)_c(2,1,1)_IPR(100,120,0)_w1.json

#rcNetwork_N(100)_gp(0,0.2)_g(0,0.1)_c(2,1,1)_IPR(100,120,0)_w1.json
netname="rcNetwork_N({})_gp({},{})_g({},{})_c({},{},{})_IPR({},{},{})_w{}.json".format(N,gpM,gpS,gM,gS,cF,cB,cS,I,P,R,w)
print(netname)

cmd_list=["../../build/release/run_rc",
          netname, # path to JSON file
          "../../data/pythonExample.txt", # path to data file
          "0", # gaussian noise on timestamps of the data mean of 0 standard deviation of 1.0
          "1", # percentage of additive noise
          "spikeLog.bin", # name of output spike file
          "potentialLog.bin", # name of output potential file
          "0", # Enable GUI (bool)
          "0", # timestep
          "0",
          # str(list(range(N*2))),
          # ">/dev/null"
               # either leave this field blank -> potential logs all the reservoir, or, any number that goes after the timestep argument will be tracked. so you can add "0", "1", "2" to track the neurons with ID 0, 1 and 2
        ]

# subprocess.call(cmd_list)

trfiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TrainTXT")) for f in fn]
tefiles=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/Datasets/N-MNIST/TestTXT")) for f in fn]

N=len(trfiles)
# for i,f in enumerate(trfiles):
trdirsp=os.path.expanduser('~/Datasets/N-MNIST/TrainRCSp')
trdirpt=os.path.expanduser('~/Datasets/N-MNIST/TrainRCPt')
tedirsp=os.path.expanduser('~/Datasets/N-MNIST/TestRCSp')
tedirpt=os.path.expanduser('~/Datasets/N-MNIST/TestRCPt')


if not os.path.exists(trdirpt):
     os.mkdir(trdirpt)
if not os.path.exists(trdirsp):
     os.mkdir(trdirsp)
if not os.path.exists(tedirsp):
     os.mkdir(tedirsp)
if not os.path.exists(tedirpt):
     os.mkdir(tedirpt)
start_time = time() 
def trainfiles(i,f):
     fsp=f.replace('TrainTXT','TrainRCSp').replace('.txt','.bin')
     fpt=f.replace('TrainTXT','TrainRCPt').replace('.txt','.bin')
     dnamesp= '/'.join(fsp.split('/')[:-1])
     dnamept= '/'.join(fpt.split('/')[:-1])
     if os.path.exists(dnamesp)==False:
          try:
               pathlib.Path(dnamesp).mkdir(parents=True, exist_ok=True)
          finally:
               sleep(0.5)
     if os.path.exists(dnamept)==False:
          try:
               pathlib.Path(dnamept).mkdir(parents=True, exist_ok=True)
          finally:
               sleep(0.5)
     cmd_list[2]=f
     # print(f)
     cmd_list[5]=fsp
     cmd_list[6]=fpt
     # print("Calling {}".format(cmd_list))
     subprocess.call(cmd_list)
     end_time=time()
     my_time=end_time-start_time
     if i>0:
          ETA = (float(N)/float(i) -1)*my_time
          ETAhr = int(ETA/3600)
          ETAmn = int(ETA/60)%60
          ETAsc = int(ETA)%60
          print("\n\nTraining  {} of {} files \n\nTime: {} ETA: {:02d}:{:02d}:{:02d} \n\n".format(i,N,my_time,ETAhr,ETAmn,ETAsc))

     # print(cmd_list)
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(trainfiles)(i,f) for (i,f) in enumerate(trfiles) )

start_time = time() 
N=len(tefiles)
# for f in enumerate(tefiles):
def testfiles(i,f):
     fsp=f.replace('TestTXT','TestRCSp').replace('.txt','.bin')
     fpt=f.replace('TestTXT','TestRCPt').replace('.txt','.bin')
     dnamesp= '/'.join(fsp.split('/')[:-1])
     dnamept= '/'.join(fpt.split('/')[:-1])
     if os.path.exists(dnamesp)==False:
          os.makedirs(dnamesp)#.mkdir(parents=True, exist_ok=True)
     if os.path.exists(dnamept)==False:
          os.makedirs(dnamept)#.mkdir(parents=True, exist_ok=True)
     cmd_list[2]=f
     cmd_list[5]=fsp
     cmd_list[6]=fpt
     subprocess.call(cmd_list)
     end_time=time()
     my_time=end_time-start_time
     if i>0:
          ETA = my_time*((float(N)/float(i))-1)
          ETAhr = int(ETA/3600)
          ETAmn = int(ETA/60)%60
          ETAsc = int(ETA)%60
          print("\n\nTesting {} out of {} files \n\nTime: {} ETA: {:02d}:{:02d}:{:02d} \n\n".format(i,N,my_time,ETAhr,ETAmn,ETAsc))

Parallel(n_jobs=num_cores)(delayed(testfiles)(i,f) for (i,f) in enumerate(tefiles))

# NOTES: If a network has too many events, it might bmanye faster to run the network clock-based

import numpy as np
import struct

def spike_reader(log):
    """
    Reads files in with a format:
        SpikeLogger Binary File Specs |
        ------------| --------
        timestamp diff (x100) | byte 0 to 4 (int32)
        delay (x100) | byte 4 to 6 (int16)
        weight (x100)| byte 6 to 7 (int8)
        potential (x100)| byte 7 to 9 (int16)
        neuron ID | byte 9 to 11 (int16)
        layer ID | byte 11 to 12 (int8)
        receptive field row index | byte 12 to 13 (int8)
        receptive field column index | byte 13 to 14 (int8)
        x coordinate | byte 14 to 15 (int8)
        y coordinate | byte 15 to 16 (int8)
    """
    with open(log,'rb') as f:
        d=f.read()
        offset=8
        i=0
        chunk_size=16
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('=ihb2h5b',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def regression_spike_parser(logname, p=28, layer=-1, N=-1):
    log=np.array(spike_reader(logname))
    tstmps = sorted(list(set(log[:,0])))
    
    t2i = { t:i for i,t in enumerate(tstmps)}
    T=len(tstmps)

    Sp = np.zeros((T,N),dtype=np.int8)

    for i,n in enumerate(log[:,0]):
        if log[i,5]<(p**2):
            continue
        Sp[t2i[n], int(log[i,5])-(p**2)]=1
    return Sp

def potential_reader(log):
    """
    Reads files in with a fromat:
        PotentialLogger Binary File Specs |
        ------------| --------
        timestamp diff (x100) | byte 0 to 4 (int32)
        potential (x100) | byte 4 to 6 (int16)
        postsynaptic neuron ID | byte 6 to 8 (int16)
    """
    with open(log,'rb') as f:
        d=f.read()
        offset=0
        i=0
        chunk_size=8
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('i2h',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def regression_potential_parser(logname, p=28,layer=-1,N=-1,ds=0):
    log=np.array(potential_reader(logname))
    log[:,0]=np.cumsum(log[:,0])//100
    if ds>0:
        log[:,0]=log[:,0]//ds
    tstmps = np.arange(np.min(log[:,0]),np.max(log[:,0])+1)
    t2i = { t:i for i,t in enumerate(tstmps)}
    
    T=len(tstmps)

    minID = p**2
    Sp = np.zeros((T,N))
    for i,n in enumerate(log[:,0]):
        tindex= t2i[n]
        nindex= log[i,2]-minID
        Sp[tindex, nindex]+=log[i,1]
    return Sp

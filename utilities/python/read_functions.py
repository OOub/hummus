import numpy as np
import struct

def sizemsg(d):
    if type(d)==int:
        if d>1e+9:
            print("Read {} GB".format(int(d/1e+9)))
        elif d>1e+6:
            print("Read {} MB".format(int(d/1e+6)))
        elif d>1e+3:
            print("Read {} KB".format(int(d/1e+3)))
        else:
            print("Read {} B".format(int(d)))


def read_spikeLog(log):
    """
    Reads files in with a fromat:
        SpikeLogger Binary File Specs |
        ------------| --------
        timestamp | byte 0 to 8 (double)
        delay | byte 8 to 12 (float)
        weight | byte 12 to 16 (float)
        potential | byte 16 to 20 (float)
        presynaptic neuron ID | byte 20 to 22 (int16)
        postsynaptic neuron ID | byte 22 to 24 (int16)
        layer ID | byte 24 to 26 (int16)
        receptive field row index | byte 26 to 28 (int16)
        receptive field column index | byte 28 to 30 (int16)
        x coordinate | byte 30 to 32 (int16)
        y coordinate | byte 32 to 34 (int16)
    """
    with open(log,'rb') as f:
        d=f.read()
        # sizemsg(len(d))
        offset=8
        i=0
        chunk_size=34
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('dfffhhhhhhh',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def read_spikeLog2(log):
    """
    Reads files in with a fromat:
        SpikeLogger Binary File Specs |
        ------------| --------
        timestamp diff (x100) | byte 0 to 2 (int16)
        delay (x100) | byte 2 to 4 (int16)
        weight (x100)| byte 4 to 5 (int8)
        potential (x100)| byte 5 to 7 (int16)
        neuron ID | byte 7 to 9 (int16)
        layer ID | byte 9 to 10 (int8)
        receptive field row index | byte 10 to 11 (int8)
        receptive field column index | byte 11 to 12 (int8)
        x coordinate | byte 12 to 13 (int8)
        y coordinate | byte 13 to 14 (int8)
    """
    with open(log,'rb') as f:
        d=f.read()
        # sizemsg(len(d))
        offset=8
        i=0
        chunk_size=14
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('=2hb2h5b',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def read_spike_np2(logname, layer=-1,entry=None, binary=False,N=-1):
    log=np.array(read_spikeLog2(logname))
    tstmps = sorted(list(set(log[:,0])))
    
    t2i = { t:i for i,t in enumerate(tstmps)}
    T=len(tstmps)


    if N==-1:
        N=len(set(log['info'][:,1]))-1
    if binary:
        Sp = np.zeros((T,N),dtype=np.int8)
    else:
        Sp = np.zeros((T,N))
    for i,n in enumerate(log[:,0]):
        if log[i,5]<(28*28):
            continue
        if binary:
            Sp[t2i[n], int(log[i,5])-(28**2)]=1
        else:
            Sp[t2i[n], int(log[i,5])-(28**2)]=log['potential'][i]
    return Sp

def read_spike_data(log, layer=-1, entry=None):
    l=read_spikeLog(log)
    T = np.zeros(len(l))
    Dl = np.zeros(len(l))
    Pt = np.zeros(len(l))
    I = np.zeros((len(l),7),dtype=np.int16)
    for i,d in enumerate(l):
        if layer<0:
            T[i]=d[0]
            Dl[i]=d[1]
            Pt[i]=d[3]
            I[i]=d[4:]

        else:
            if layer==d[6]:
                T[i]=d[0]
                Dl[i]=d[1]
                Pt[i]=d[3]
                I[i]=d[4:]
    if entry is not None:
        return {"timestamp": T, "delay": Dl, "potential": Pt, "info": I, "entry": np.ones(len(l))*entry}

    return {"timestamp": T, "delay": Dl, "potential": Pt, "info": I, }

def read_spike_np(logname, p=28,layer=-1,entry=None, N=-1,binary=False):
    log=read_spike_data(logname,layer,entry)
    tstmps = sorted(list(set(log['timestamp'])))
    t2i = { t:i for i,t in enumerate(tstmps)}
    T=len(tstmps)

    if N==-1:
        N=len(set(log['info'][:,1]))-1
        
    if binary:
        Sp = np.zeros((T,N),dtype=np.int8)
    else:
        Sp = np.zeros((T,N))
    minID=p**2
    for i,n in enumerate(log['timestamp']):
        if log['info'][i,1]==0:
            continue
        if binary:
            Sp[t2i[n], log['info'][i,1]-minID]=1
        else:
            Sp[t2i[n], log['info'][i,1]-minID]=log['potential'][i]
    return Sp


def read_potentialLog(log):
    """
    Reads files in with a fromat:
        PotentialLogger Binary File Specs |
        ------------| --------
        timestamp | byte 0 to 8 (double)
        potential | byte 8 to 12 (float)
        postsynaptic neuron ID | byte 12 to 14 (int16)
    """
    with open(log,'rb') as f:
        d=f.read()
        # sizemsg(len(d))
        offset=0
        i=0
        chunk_size=8
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('i2h',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def read_pot_np(logname, p=28,layer=-1,entry=None, binary=False,N=-1,ds=0):
    # import ipdb;ipdb.set_trace()
    log=np.array(read_potentialLog(logname))
    log[:,0]=np.cumsum(log[:,0])//100
    tstmps = sorted(list(set(log[:,0])))
    t2i = { t:i for i,t in enumerate(tstmps)}
    T=len(tstmps)

    if N==-1:
        N=len(set(log[:,2]))
    minID = p**2
    if binary:
        if ds>0:
            Sp = np.zeros(((T//ds)+1,N),dtype=np.int8)
        else:
            Sp = np.zeros((T,N),dtype=np.int8)
    else:
        if ds>0:
            Sp = np.zeros(((T//ds)+1,N))
        else:
            Sp = np.zeros((T,N))
    for i,n in enumerate(log[:,0]):
        tindex= t2i[n]#//ds
        if ds>0:
            tindex= t2i[n]//ds
        nindex= log[i,2]-minID
        if binary:
            Sp[tindex, nindex]+=1
        else:
            Sp[tindex, nindex]+=log[i,1]
    return Sp

def read_pot_np_ds(logname, p=28,layer=-1,entry=None, binary=False,N=-1,ds=0):
    # import ipdb;ipdb.set_trace()
    log=np.array(read_potentialLog(logname))
    log[:,0]=np.cumsum(log[:,0])//100
    if ds>0:
        log[:,0]=log[:,0]//ds
    tstmps = np.arange(np.min(log[:,0]),np.max(log[:,0])+1)
    # tstmps = sorted(list(set(log[:,0])))
    t2i = { t:i for i,t in enumerate(tstmps)}
    
    T=len(tstmps)

    if N==-1:
        N=len(set(log[:,2]))
    minID = p**2
    if binary:
        Sp = np.zeros((T,N),dtype=np.int8)
    else:
        Sp = np.zeros((T,N))
    for i,n in enumerate(log[:,0]):
        tindex= t2i[n]
        nindex= log[i,2]-minID
        if binary:
            Sp[tindex, nindex]+=1
        else:
            Sp[tindex, nindex]+=log[i,1]
    return Sp

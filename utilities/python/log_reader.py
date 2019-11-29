import numpy as np
import struct

def spike_reader(log, efficient_format):
    """
    Reads spike_logger files

    EFFICIENT FORMAT
        timestamp difference (x100) | byte 0 to 4 (int32)
        delay (x100) | byte 4 to 6 (int16)
        weight (x100)| byte 6 to 7 (int8)
        potential (x100)| byte 7 to 9 (int16)
        presynaptic neuron ID | byte 9 to 11 (int16)
        postsynaptic neuron ID | byte 9 to 11 (int16)
        layer ID | byte 11 to 12 (int8)

    FULL FORMAT
        raw timestamp | byte 0 to 8 (double)
        delay | byte 8 to 10 (float)
        weight | byte 10 to 12 (float)
        potential (x100)| byte 7 to 9 (int16)
        presynaptic neuron ID | byte 9 to 11 (int16)
        postsynaptic neuron ID | byte 9 to 11 (int16)
        layer ID | byte 11 to 12 (int8)


    """
    with open(log,'rb') as f:
        d=f.read()
        offset=8
        i=0
        chunk_size=15
        l=[]
        while ((len(d)-offset)>=(i+1)*chunk_size):
            l.append(list(struct.unpack('=ihb2h4b',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
            i+=1
    return l

def potential_reader(log):
    """
    Reads potential_logger files with a format:
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

def weight_maps_reader(log):
    print("not implemented yet")

def myelin_plasticity_reader(log):
    print("not implemented yet")

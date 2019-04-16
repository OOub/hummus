import numpy as np
import struct

def spikeReader(log):
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

def potentialReader(log):
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
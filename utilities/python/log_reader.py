import numpy as np
import struct

def spike_reader(log, efficient_format):
    """
    Reads spike_logger files
    
    PARAMETERS
    ----------
    log - filename
    efficient_format - true for efficient format and false for full format
    
    EFFICIENT FORMAT BINARY SPECS
    -----------------------------
    timestamp difference (x100) | int32
    delay (x100)                | int16
    weight (x100)               | int8
    potential (x100)            | int16
    presynaptic neuron ID       | int16
    postsynaptic neuron ID      | int16
    layer ID                    | int8

    FULL FORMAT BINARY SPECS
    ------------------------
    raw timestamp          | double
    delay                  | float
    weight                 | float
    potential (x100)       | int16
    presynaptic neuron ID  | int16
    postsynaptic neuron ID | int16
    layer ID               | int8
    """
    if efficient_format:
        with open(log,'rb') as f:
            d=f.read()
            offset=8
            i=0
            chunk_size=14
            l=[]
            while ((len(d)-offset)>=(i+1)*chunk_size):
                l.append(list(struct.unpack('=ihb3hb',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
                i+=1
        return l
    else:
        with open(log,'rb') as f:
            d=f.read()
            offset=8
            i=0
            chunk_size=23
            l=[]
            while ((len(d)-offset)>=(i+1)*chunk_size):
                l.append(list(struct.unpack('=d2f3hb',d[i*chunk_size+offset:(i+1)*chunk_size+offset])))
                i+=1
        return l
    

def potential_reader(log):
    """
    Reads potential_logger files
    
    PARAMETERS
    ----------
    log - filename
    
    BINARY SPECS
    ------------
    timestamp diff (x100)  | int32
    potential (x100)       | int16
    postsynaptic neuron ID | int16
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

def myelin_plasticity_reader(log):
    """
    Reads myelin_plasticity_logger files
    
    PARAMETERS
    ----------
    log - filename
    
    BINARY SPECS
    ------------
    bit_size               | int16
    timestamp diff (x100)  | int32
    postsynaptic neuron ID | int16
    
    For all active input neurons in learning epoch ((bit_size-8)/9)
        delta_t (x100) | int32
        neuron_id      | int16
        delay (x100)   | int16
        weight (x100)  | int16
    """
    
    print("not implemented yet")

def weight_maps_reader(log):
    """
    Reads weight_maps files
    
    PARAMETERS
    ----------
    log - filename
    
    BINARY SPECS
    ------------
    bit_size  | int16
    neuron_id | int16
    
    For all input neurons ((bit_size-4)/8)
        weight | double
    """
    
    print("not implemented yet")

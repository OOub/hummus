# dataset_converter.py 

# Created by Omar Oubari.
# Email: omar.oubari@inserm.fr
# Last Version: 22/08/2019

# Information: code used to convert popular datasets into the eventstream (.es) format accepted by the Hummus spiking neural network simulator. 
# For more details on the .es format: https://github.com/neuromorphic-paris/event_stream

import os
import loris
import numpy as np
import multiprocessing
from struct import unpack, pack
from joblib import Parallel, delayed

####### BATCH CONVERSION OF WHOLE DATASET USING PARALLELISATION #######
def batch_ncar_to_es(ncar_directory_in,ncar_directory_out):
    train_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(ncar_directory_in,"train")) for f in fn]
    test_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(ncar_directory_in,"test")) for f in fn]

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(ncar_to_es)(f_in,os.path.join(ncar_directory_out,"train",f_in.split("/")[-2],f_in.split("/")[-1].split('.')[0]+'.es')) for f_in in train_files)
    Parallel(n_jobs=num_cores)(delayed(ncar_to_es)(f_in,os.path.join(ncar_directory_out,"test",f_in.split("/")[-2],f_in.split("/")[-1].split('.')[0]+'.es')) for f_in in test_files)

def batch_poker_to_es(poker_directory_in,poker_directory_out):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(poker_directory_in) for f in fn]

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(poker_to_es)(f_in,os.path.join(poker_directory_out,f_in.split("/")[-2],f_in.split("/")[-1].split('.')[0]+'.es')) for f_in in files)

def batch_nmnist_to_es(nmnist_directory_in,nmnist_directory_out):
    train_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(nmnist_directory_in,"Train")) for f in fn]
    test_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(nmnist_directory_in,"Test")) for f in fn]

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(nmnist_to_es)(f_in,os.path.join(nmnist_directory_out,"Train",f_in.split("/")[-2],f_in.split("/")[-1].split('.')[0]+'.es')) for f_in in train_files)
    Parallel(n_jobs=num_cores)(delayed(nmnist_to_es)(f_in,os.path.join(nmnist_directory_out,"Test",f_in.split("/")[-2],f_in.split("/")[-1].split('.')[0]+'.es')) for f_in in test_files)

####### METHODS TO CONVERT FILES FROM POPULAR NEUROMORPHIC DATASETS INTO THE ES FORMAT #######
def ncar_to_es(filepath_in, filepath_out, verbose=False):
    if filepath_in.split('.')[1] == 'dat':
        # create the output directory if it doesn't exist
        basepath_out = os.path.dirname(filepath_out)
        if not os.path.exists(basepath_out):
            os.makedirs(basepath_out)
            
        # read file
        data = loris.read_file(filepath_in)
        
        # parse data according to the N-CAR specs
        data['width'] = 64
        data['height'] = 56

        for event in data['events']:
            event[2] = 55 - event[2]

        # write to es
        loris.write_events_to_file(data, filepath_out)
        
    else:
        if verbose:
            print(filepath_in, "is not an accepted file")
        
def poker_to_es(filepath_in, filepath_out, verbose=False):
    """ Converts a file from the POKER-DVS dataset into the .es format with dimensions 35x35"""

    if filepath_in.split('.')[1] == 'dat':

        # create the output directory if it doesn't exist
        basepath_out = os.path.dirname(filepath_out)
        if not os.path.exists(basepath_out):
            os.makedirs(basepath_out) 

        # read file
        data = loris.read_file(filepath_in)

        # parse data according to the POKER-DVS specs
        data['width'] = 35
        data['height'] = 35

        for event in data['events']:
            event[2] = 239 - event[2]

        # write to es
        loris.write_events_to_file(data, filepath_out)
    else:
        if verbose:
            print(filepath_in, "is not an accepted file")

def nmnist_to_es(filepath_in, filepath_out, verbose=False):
    """ Converts a file from the N-MNIST dataset into the .es format with dimensions 34x34"""

    if filepath_in.split('.')[1] == 'bin':

        # create the output directory if it doesn't exist
        basepath_out = os.path.dirname(filepath_out)
        if not os.path.exists(basepath_out):
            os.makedirs(basepath_out) 

        # read file
        events = read_bin(filepath_in)

        # build the dictionary for .es compatibility
        data = {'type':'dvs','width':34,'height':34,'events':events}

        # write to es
        loris.write_events_to_file(data, filepath_out)
    else:
        if verbose:
            print(filepath_in, "is not an accepted file")


def gesture_to_es(filepath_in, filepath_out, verbose=False):
    """ Converts a file from the DVS128 GESTURE dataset into the .es format with dimensions 128x128"""

    if filepath_in.split('.')[1] == 'aedat':

        # create the output directory if it doesn't exist
        basepath_out = os.path.dirname(filepath_out)
        if not os.path.exists(basepath_out):
            os.makedirs(basepath_out) 

        # read file
        events = read_aedat(filepath_in)

        for event in events:
            event[2] = 127 - event[2]

        # build the dictionary for .es compatibility
        data = {'type':'dvs','width':128,'height':128,'events':events}

        # write to es
        loris.write_events_to_file(data, filepath_out)
    else:
        if verbose:
            print(filepath_in, "is not an accepted file")

####### METHOD TO READ BIN - Adapted from: https://github.com/gorchard/event-Python/blob/9fe0f12e1108c48e1e7e62b5529ed1f5518f21b9/eventvision.py#L532  #######
def read_bin(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    zipped_data = list(zip(all_ts[td_indices],all_x[td_indices],all_y[td_indices],all_p[td_indices]))
    formatted_data = np.array(zipped_data, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('is_increase', '?')])
    return formatted_data

####### METHODS TO READ AEDAT - Adapted from code provided by Jean-Matthieu Maro #######
def peek(f, length=1):
    pos = f.tell()
    data = f.read(length)
    f.seek(pos)
    return data

def readHeader(file):
    line = file.readline()
    if line != b'#!AER-DAT3.1\r\n':
        print('Wrong format: not AER-DAT3.1')
    while peek(file) == b'#':
        line = file.readline()
        # print(line)

def readEventCommonHeader(file):
    """ read common header of aedat 3.1 format """
    eventType = unpack('H',file.read(2))[0]
    eventSource = unpack('H',file.read(2))[0]
    eventSize = unpack('I', file.read(4))[0]
    eventTSOffset = unpack('I', file.read(4))[0]
    eventTSOverflow = unpack('I', file.read(4))[0]
    eventCapacity = unpack('I', file.read(4))[0]
    eventNumber = unpack('I', file.read(4))[0]
    eventValid = unpack('I', file.read(4))[0]
    if eventType != 1:
        print("Not Polarity Events!")
    return (eventType, eventSource, eventSize, eventTSOffset, eventTSOverflow, eventCapacity, eventNumber, eventValid)

def readPolarityEvent(file):
    """ read a polarity event in aedat 3.1 format """
    data = unpack('I', file.read(4))[0]
    timestamp = unpack('I', file.read(4))[0]
    validity = data & 0x00000001
    if validity:
        x = ( data >> 17 ) & 0x00001FFF
        y = ( data >> 2 ) & 0x00001FFF
        polarity = ( data >> 1 ) & 0x00000001
        return (timestamp, x, y , polarity)
    else:
        return None

def assessNumberOfPolarityEventsInFile(file):
    totalPolarityEventsInFile = 0
    while peek(file):
        commonHeader = readEventCommonHeader(file)
        eventNumber = commonHeader[6]
        if commonHeader[0] == 1: # if these are Polarity Events
            totalPolarityEventsInFile += commonHeader[7]
            for ii in np.arange(eventNumber):
                data = unpack('I', file.read(4))[0] # not stored yet
                timestamp = unpack('I', file.read(4))[0] # not stored yet
            if commonHeader[7] != eventNumber:
                print('unvalid events!')

    print('Number of polarity events: {0}'.format(totalPolarityEventsInFile))
    return totalPolarityEventsInFile

def readPolarityEventsInFile(totalPolarityEventsInFile, file):
    """ read events from a file and load them into numpy arrays """
    k = 0
    ts = np.zeros(totalPolarityEventsInFile, dtype = np.int64)
    x = np.zeros(totalPolarityEventsInFile, dtype = np.int8)
    y = np.zeros(totalPolarityEventsInFile, dtype = np.int8)
    p = np.zeros(totalPolarityEventsInFile, dtype = np.bool)

    while peek(file):
        commonHeader = readEventCommonHeader(file)
        eventNumber = commonHeader[6]
        if commonHeader[0] == 1: # if these are Polarity Events
            for ii in np.arange(eventNumber):
                e = readPolarityEvent(file)
                if e != None:
                    ts[k], x[k], y[k], p[k] = e
                    k += 1

            if commonHeader[7] != eventNumber:
                print('unvalid events!')
        else:
            print('Not Polarity Events! (not handled)')
    return ts, x, y, p

def read_aedat(aedatfile):
    print('Reading aedat file... (' + aedatfile + ')')
    file = open(aedatfile,'rb')

    # read file header
    readHeader(file)

    # first assess the number of events in the file
    pos = file.tell()
    totalPolarityEventsInFile = assessNumberOfPolarityEventsInFile(file)
    file.seek(pos)

    # put events in lists
    ts, x, y, pol = readPolarityEventsInFile(totalPolarityEventsInFile, file)

    file.close()

    zipped_data = list(zip(ts,x,y,pol))
    formatted_data = np.array(zipped_data, dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('is_increase', '?')])
    return formatted_data

if __name__ == "__main__":
    batch_ncar_to_es("/Users/omaroubari/Datasets/N-CARS/", "/Users/omaroubari/Datasets/es_N-CARS/")

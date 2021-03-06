from functools import reduce
import os
import tensorflow as tf
import nsd_access as nsda
import re
import numpy as np
import random
import h5py

def BufferedLoader(nsd_root, subjects, bufsize, shape=(327684,), format='fsaverage', type="betas_fithrf_GLMdenoise_RR", shufflefiles=True, shuffledata=True):
    '''
    nsd_root    -   path to root of nsd dataset
    type        -   preprocessing type
    format      -   format of betas to load
    subject     -   subject identifier strings (list)
    bufsize     -   number of samples to (minimally) initialize loader with

    creates a tensorflow dataset that sequentially loads session files into memory.
    Loads $bufsize files at once and uses up all trials before loading new trials

    Probably should be used with care if data from multiple participants is used to train a model since
    there are no guarantees that each time the buffer is filled with trials from all participants
    '''
    nsdaccess = nsda.NSDAccess(nsd_root)

    def load_files(subj_sess):
        '''
        loads betas from all specified files and concatenates betas into a single array
        '''
        betas = list()
        for (subj, sess) in subj_sess:
            b = nsdaccess.read_betas(subj, sess, data_type=type, data_format=format)
            b = b.T
            betas.append(b)
        return np.vstack(betas)

    def nsd_generator():
        
        # get all available session files for participants
        subj_sessions = list()
        betafolder = nsdaccess.nsddata_betas_folder
        getsessionfiles = re.compile("[rl]h.betas_session\d\d.mgh")
        getnumber = re.compile("\d\d")
        for s in subjects:
            datafolder = os.path.join(betafolder, s, format, type)
            files = os.listdir(datafolder)
            allfiles = ''.join(files)
            sessionfiles = getsessionfiles.findall(allfiles)
            sessionindices = np.unique([int(num) for num in getnumber.findall(''.join(sessionfiles))])
            print(f"found {len(sessionindices)} sessions for subject {s}")
            ss = [(s, sess) for sess in sessionindices]
            subj_sessions += ss
        
        if shufflefiles:
            random.shuffle(subj_sessions)

        while subj_sessions:
            numfiles = len(subj_sessions)
            if numfiles > bufsize:
                betas = load_files(subj_sessions[:bufsize])
                subj_sessions = subj_sessions[bufsize:]
            else:
                betas = load_files(subj_sessions)
                subj_sessions = list()

            numbetas = betas.shape[0]
            indices = list(range(numbetas))
            
            if shuffledata:
                random.shuffle(indices)
            for i in indices:
                yield betas[i]

    return nsd_generator
        
def hdf5_generator(path, dataset):
    with h5py.File(path, 'r') as f:
        num_datapoints = f[dataset].shape[0]
        for i in range(num_datapoints):
            yield f[dataset][i,:]

if __name__ == "__main__":
    path = '/home/daniel/Documents/Masterarbeit/NSD/nsddata_betas/ppdata/subj02/fsaverage/betas_fithrf_GLMdenoise_RR/subj02.hdf5'
    dataset = 'subj02'

    ds = tf.data.Dataset.from_generator(hdf5_generator, args=[path, dataset], output_types=tf.float32)

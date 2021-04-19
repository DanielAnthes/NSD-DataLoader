import nsd_access as nsda
import pandas as pd
from os import path as op
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from math import ceil
from random import sample
import os


class NSDLoader:
    '''
    Data Loader for the nsd dataset based on the nsd_access package:
    https://github.com/tknapen/nsd_access
    Allows for loading beta values, captions and images from the nsd dataset and splitting them
    into training and test sets. Optionally creates an iterator for loading the data in batches.
    '''

    def __init__(self, nsd_loc, data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage'):
        '''
        INPUTS:
            nsd_loc: path to the roor folder of the nsd dataset, file structure is assumed to be as in the original dataset
            data_type: types of betas to load
            data_format: format of betas to load
        '''
        self.nsda = nsda.NSDAccess(nsd_loc)
        self.data_type = data_type
        self.data_format = data_format

    def extract_Coco_Annotations(self, coco_info):
        '''
        For each 73k index read_image_coco_info returns a list of dictionaries, 
        this helper function extracts all captions into a single list
        
        INPUTS:
            coco_info: information about one stimulus in the coco dataset, as returned by read_image_coco_info (nsda)
        RETURNS:
            captions: a list of captions labelling the stimulus
        '''
        captions = list()
        for info_dict in coco_info:
            captions.append(info_dict['caption'])
        return captions
    
    def get_data_by_trial(self, subj, sess, trial, load_images=True):
        '''
        INPUTS:
            subj:   subject identifier string e.g. 'subj01'
            sess:   session number, int (starting to count from 1)
            trial:  list of trials to get, by index. TODO check index base
        
        RETURNS:
            tuple (stimulus, captions, betas)
            with
                stimulus: image presented at trial(s), numpy array
                captions: list of captions from COCO, or list of list for multiple data points
                betas: beta values for trial(s), numpy array
        '''
        behav_data = self.nsda.read_behavior(subj, sess, trial_index=trial) # nsda claims to count sessions starting from 0, but this does not seem to be the case
        im_id_73_info = behav_data['73KID']
        idx_73k = im_id_73_info.to_list()
        betas = self.nsda.read_betas(subj, sess, trial_index=trial, data_type=self.data_type, data_format=self.data_format)
        
        if len(trial) == 1:
            coco_return = self.nsda.read_image_coco_info(idx_73k)
            captions = self.extract_Coco_Annotations(coco_return)
        else:
            captions = [self.extract_Coco_Annotations(x) for x in self.nsda.read_image_coco_info(idx_73k)]

        if not load_images:
            return betas, captions
        
        else:
            # to get images 73k indices must be sorted
            idx_73k = np.array(idx_73k)

            idx = np.argsort(idx_73k)

            # hdf5 library needs indices to be STRICTLY increasing, so remove duplicates and only load each image once

            unique_id = np.unique(idx_73k[idx])
            im = self.nsda.read_images(unique_id, show=False)
            # associate loaded images with correct 73k index
            if len(trial) == 1:
                im = im[0]
            else:
                imshape = im.shape[1:]
                images = np.zeros((len(idx_73k), *imshape), dtype=np.uint8)
                for i, im_index in enumerate(idx_73k):
                    impos = np.where(unique_id==im_index)[0][0]
                    images[i,:,:,:] = im[impos,:,:,:]
                im = images
            return betas, captions, im

    def get_data_info(self, verbose=False):
        '''
        gather info about available data
        optionally print summary

        RETURNS:
            nested dictionary:
                key:    subject code
                value:  dict:
                        key: session
                        value = number of trials 
        '''
        # list participant folders:
        beta_dir = os.listdir(self.nsda.nsddata_betas_folder)
        r = re.compile("subj\d\d") # subject folders have shape subjxx, with xx two digits e.g. subj01
        subjects = [x for x in beta_dir if r.match(x)] # list of all subjects for which we have betas locally
        data_per_subj = dict()
        for s in subjects:
            
            if verbose:
                print(s)
            
            behavior = pd.read_csv(self.nsda.behavior_file.format(subject=s), delimiter='\t') # open info file for subject
            sessions = behavior["SESSION"].unique() # get all session ids
            
            if verbose:
                print(f"{len(sessions)} sessions available")
            
            trials = list()
            total_trials = 0
            
            for sess in sessions:
                sess_trials = behavior["TRIAL"][behavior["SESSION"] == sess] # .unique() # filter for current session and return all unique trial numbers (there should not be any doubles in the first place)
                trials.append(len(sess_trials))
                total_trials += len(sess_trials)
            
            if verbose:
                print(f"{total_trials} trials available")

            session_dict = dict()
            session_dict.update(zip(sessions, trials))
            data_per_subj[s] = session_dict

        return data_per_subj

    def get_stimulus_ids(self, subset="shared", participant=""):
        '''
        INPUT:
            subset - one of "shared", "unique", "all": filter dataset to return only shared stimuli, only stimuli not seen by every participant, or all stimuli
            participant - optional: filter for stimuli shown to participant

        OUTPUT:
            cocoIDs of selected stimuli, pandas Series object, index is 73k index in [0 72999]
        '''
        if not hasattr(self.nsda, 'stim_descriptions'): # nsda only loads stim_description file to memory once, if this has not happened yet do it now
                self.nsda.stim_descriptions = pd.read_csv(
                    self.nsda.stimuli_description_file, index_col=0)
        stim_descr = self.nsda.stim_descriptions
        
        # filter for participant
        if len(participant) > 0:
            p_number = int(participant[-2:])
            col = stim_descr.columns[7 + p_number]
            print(f"filtering dataset for : {col}")
            mask = stim_descr[col] == 1
            stim_descr = stim_descr[mask]
            print(f"number of stimuli available for {col}: {len(stim_descr)}")

        # get  image ids
        if subset == "shared":
            coco_ids = stim_descr[stim_descr['shared1000']==True]['cocoId']
        elif subset == "all":
            coco_ids = stim_descr['cocoId']
        elif subset == "unique":
            coco_ids = stim_descr[stim_descr['shared1000']==False]['cocoId']
        else:
            raise Exception("unknown subset parameter")

        print(f"after filtering for: {subset}, {len(coco_ids)} stimuli are available")
        # add 73K index as column explicitly
        data = pd.DataFrame(coco_ids).assign(ID73K=coco_ids.index.to_numpy() + 1) # TODO check for off by one error, 
        return data

    def get_stim_reps_per_subj(self, subj): # TODO untested
        '''
        INPUT:
            subj - integer value 1..8 indicating the subject for which to get data
        
        RETURNS:
            rep_frame: two column pandas dataframe with '73K' the index of the stimulus, 'n_reps' number of repetitions for this subject
                    only contains entries where n_reps > 0
        '''
        if not hasattr(self.nsda, 'stim_descriptions'): # nsda only loads stim_description file to memory once, if this has not happened yet do it now
            self.nsda.stim_descriptions = pd.read_csv(
                self.nsda.stimuli_description_file, index_col=0)
        stim_descr = self.nsda.stim_descriptions
        # filter for columns relevant to subject, starting from column 16 each subject has 3 columns containing either 0 or the trial id of the repretition
        stim_data_subj = stim_descr.iloc[:,16+(subj-1)*3:16+(subj-1)*3+3]
        stim_reps_subj = stim_data_subj.gt(0).sum(axis=1) # finds all values greater than zero and sums over columns
        stim_reps_subj = stim_reps_subj[stim_reps_subj.gt(0)] # filter for images that were shown to the subject
        stim_reps_73k = stim_reps_subj.index.to_series() # get 73k indices of images
        rep_frame = pd.concat([stim_reps_73k, stim_reps_subj], axis=1)
        rep_frame.columns = ['73K', 'n_reps']
        return rep_frame

    def trialID_to_sess_trial(self, data_info, subj, trialID): # TODO untested, unused
        '''
        INPUT:
            data_info - dictionary as constructed by get_data_info
            subj - subject string
            trialID - trial id in 1..30000
        
        RETURNS:
            sess - session number
            num_trial - index of trial in session
        '''
        subj_info = data_info[subj]
        trial_counts = subj_info.values

        sess = 1
        trial_count = trial_counts[0]

        while trialID > trial_count:
            trial_count += trial_counts[sess]
            sess += 1
        
        num_trial = trialID - sum(trial_counts[:sess])
        return sess, num_trial
    
    def calculate_session_index(self, trialinfo): # TODO untested
        '''
        INPUTS:
            trialinfo: dataframe with information about trials to collect, as created by trials_for_stim
        RETURNS:
            updated trialinfo dataframe with added column "SESS_IDX" containing 0 based index of trial within a session

        Data is organized in sessions, runs and trials:
        each full session consists of 12 runs where even runs have trial numbers 1:62, odd runs have trials 1:63
        '''
        prev_run = trialinfo["RUN"] - 1
        trial = trialinfo["TRIAL"]
        # calculate how many trials passed in previous runs in the current session
        num_even = prev_run // 2
        num_odd = num_even.apply(lambda x: x if x % 2 == 0 else x+1)
        prev_trials = num_even * 62 + num_odd * 63
        sess_idx = prev_trials + trial - 1 # subtract 1 for zero based index within session
        trialinfo = trialinfo.assign(SESS_IDX=sess_idx) # add output column
        return trialinfo
        


    def create_image_split(self, data_fraction=1, test_fraction=.2, subset="shared", participant=""):
        '''
        randomly splits all trials into a training and a test set so that each stimulus only occurs in one set
        
        INPUTS:
            subset          - string in "shared", "unique", "all" filters stimuli by either images seen by all participants, images not seen by all participants, all images
            data_fraction   - percentage of available stimuli to select
            test_fraction   - fraction of selected stimuli to hold out for testing 
                                (e.g. for 100 trials with data_fraction=.5 and test_fraction=.5 the function will return a training and a test set with 25 trials each)
            participant    - optional: give the name of a participant to ensure that the sampled stimuli were shown to that participant

        RETURNS:
            pandas dataframes with stimulus information. Split into training and test set
        '''
        ids = self.get_stimulus_ids(subset, participant) # filtering for participant happens here if participant flag is set
        ids = ids.sample(frac=data_fraction, replace=False) # randomly sampling all datapoints in pandas is equivalent to shuffle, so with frac=1 returns full shuffled dataset
        
        num_ids = len(ids)
        num_test = ceil(num_ids * test_fraction)
        num_train = num_ids - num_test
        
        
        train_stimuli = ids.iloc[0:num_train]
        test_stimuli = ids.iloc[num_train:]

        return train_stimuli, test_stimuli

    def trials_for_stim(self, subjects, id_frame):
        '''
        returns a list of all trials for a given participant in which the given stimuli were shown

        INPUTS:
            subjects - list of subjects for which to retrieve trial data
            id_frame - pandas frame with stimulus indices as produced by create_imgage_split
        
        RETURNS:
            pandas dataframe containing information needed to load data, for all trials in which stimuli in id_frame were shown
        '''
        # initialize dataframe to return
        trial_info = pd.DataFrame()
        for subj in subjects:
            # get subject information
            behaviour = pd.read_csv(self.nsda.behavior_file.format(
            subject=subj), delimiter='\t')
            stim_behav = behaviour[behaviour['73KID'].isin(id_frame['ID73K'].to_list())] # filter for stimuli to be selected
            stim_behav.assign(SUBJECT=subj)
            # keep relevant columns
            stim_behav = stim_behav[["SUBJECT", "SESSION", "RUN", "TRIAL", "73KID"]]
            trial_info = trial_info.append(stim_behav)
        return trial_info
    
    def load_data(self, trialinfo, load_captions=True, load_imgs=True):
        '''
        loads data for trials specified in pandas dataframe.
        
        INPUTS:
        trialinfo - Dataframe must have the following columns:
            SUBJECT - subject the datapoint belongs to
            SESSION - session the datapoint belongs to
            SESS_IDX - 0 based index of trial in session as created by calculate_session_index, NOT trial in run

        RETURNS:
            betas: numpy array containing betas for all loaded trials, shape depends on data format
            captions: list of captions describing stimulus used in each trial (list of lists for multiple trials)
            ims (optional): numpy array containing all images used in the loaded trials. Shape (trials, 425, 525, 3)
        '''
        # TODO try to limit access to coco annotation file to improve performance (~ 1.3 seconds per annotation access since file is loaded into memory)
        # alternatively adjust nsd_access package so that annotations are kept in memory
        # TODO also return subject for each datapoint?
        betas = list()
        ims = list()
        captions = list()

        subjects = trialinfo["SUBJECT"].unique()
        for i in range(len(subjects)):
            subj = subjects[i]
            subj_string = f"subj{str(subj).zfill(2)}" # create subject string of format subjAA where AA is zero padded subj number
            sessions = (trialinfo["SESSION"][trialinfo["SUBJECT"] == subj]).unique()
            for s in sessions:
                print(f"SESSION {s}")
                indices = trialinfo["SESS_IDX"][(trialinfo["SUBJECT"]==subj) & (trialinfo["SESSION"]==s)]
                if len(betas) == 0:
                    if load_imgs:
                        betas, captions, ims = self.get_data_by_trial(subj_string, s, indices.to_list(), load_images=True)
                    else:
                        betas, captions = self.get_data_by_trial(subj_string, s, indices.to_list(), load_images=False)
                else:
                    if load_imgs:
                        b, c, im = self.get_data_by_trial(subj_string, s, indices.to_list(), load_images=True)
                        captions += c
                        betas = np.concatenate((betas, b), axis=1)
                        ims = np.concatenate((ims, im), axis=0)
                    else:
                        b, c = self.get_data_by_trial(subj_string, s, indices.to_list(), load_images=False)
                        captions += c
                        betas = np.concatenate((betas, b), axis=1)

        if load_imgs:
            return betas, captions, ims
        else:
            return betas, captions

        returnvals = [betas]
        
        if load_captions:
            returnvals.append(captions)

        if load_imgs:
            returnvals.append(ims)

        return tuple(returnvals)

    def load_batch_data(self, trialinfo, batchsize, load_imgs=True, load_captions=True):
        '''
        load data in batches to avoid memory overflow

        INPUTS:
            trialinfo: pandas dataframe containing the trials to load. Must include the fields
                    SUBJECT, SESSION, SESS_IDX
            batchsize: integer value, denotes maximum number of trials to load per batch (may be less for last batch)
        RETURNS:
            data_iter: iterator object that can be used to load the data
        '''
        num_trials = len(trialinfo)
        slice_indices = np.arange(batchsize, num_trials, batchsize)
        indices = np.split(np.array(range(num_trials)), slice_indices)

        for i in indices:
            data = trialinfo.iloc[i]
            yield self.load_data(data, load_imgs=load_imgs)

    def load_batch_data_with_info(self, trialinfo, batchsize, load_imgs=True, load_captions=True):
        '''
        load data in batches, return additional information about trials
        '''
        num_trials = len(trialinfo)
        slice_indices = np.arange(batchsize, num_trials, batchsize)
        indices = np.split(np.array(range(num_trials)), slice_indices)

        for i in indices:
            data = trialinfo.iloc[i]
            yield (self.load_data(data, load_imgs=load_imgs), data)



# test cases

if __name__ == "__main__":
    nsdl = NSDLoader(ROOT)
    train_stimuli, test_stimuli = nsdl.create_image_split(subset="all", participant="subj04")
    print(train_stimuli.shape)
    print(test_stimuli.shape)
    '''
    trialdata = nsdl.trials_for_stim(['subj01','subj02'], train_stimuli)
    trialinf = nsdl.calculate_session_index(trialdata)    
    
    trialinf = trialinf[trialinf["SESSION"].isin([1,2])] # exclude data for which no betas are available locally


    data_iter = nsdl.load_batch_data(trialinf, 100, load_imgs=True)

    for b,c,i in data_iter:
        print(f"betas: {b.shape}\ncaptions: {len(c)}\nimages: {i.shape}")
    '''

    stim_descr = nsdl.nsda.stim_descriptions
    unique = stim_descr[stim_descr["shared1000"] == False]
    subjcols = unique.columns[8:16]
    unique[subjcols].apply(lambda x: np.sum(x), axis=1).max()

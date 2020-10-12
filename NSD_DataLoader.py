import nsd_access as nsda
import pandas as pd
from os import path as op
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from math import ceil
from random import sample

ROOT = op.join("..", "nsd") # root of the data folder, relative to this script. should probably be absolute in final script

class NSDLoader:

    def __init__(self, nsd_loc, data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage'):
        self.nsda = nsda.NSDAccess(nsd_loc)
        self.data_type = data_type
        self.data_format = data_format

    def extract_Coco_Annotations(self, coco_info):
        '''
        For each 73k index read_image_coco_info returns a list of dictionaries, 
        this helper function extracts all captions into a single list
        '''
        captions = list()
        for info_dict in coco_info:
            captions.append(info_dict['caption'])
        return captions
    
    def get_info_by_trial(self, subj, sess, trial):
        '''
        PARAMS:
            subj:   subject identifier string e.g. 'subj01'
            sess:   session number, int (starting to count from 1)
            trial:  trial number to get, by index. TODO check index base
        
        RETURNS:
            tuple (stimulus, captions, betas)
            with
                stimulus: image presented at trial
                captions: list of captions from COCO
                betas: beta values for trial
        '''
        behav_data = self.nsda.read_behavior(subj, sess, trial_index=trial)
        im_id_73_info = behav_data['73KID']
        idx_73k = im_id_73_info[trial].to_list()
        betas = self.nsda.read_betas(subj, sess, trial_index=trial, data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage')
        
        if len(trial) == 1:
            coco_return = self.nsda.read_image_coco_info(idx_73k)
            captions = self.extract_Coco_Annotations(coco_return)
        else:
            captions = [self.extract_Coco_Annotations(x) for x in self.nsda.read_image_coco_info(idx_73k)]

        # to get images 73k indices must be sorted
        idx_73k = np.array(idx_73k)
        idx = np.argsort(idx_73k)
        reverse_idx = np.argsort(idx)
        im = self.nsda.read_images(idx_73k[idx], show=False)
        im = im[reverse_idx]

        if len(trial) == 1:
            im = im[0]

        return captions, betas, im

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

    def get_stimulus_ids(self, shared=False):
        '''
        INPUT:
            shared - filter by stimuli shown to all participants

        OUTPUT:
            cocoIDs of selected stimuli, pandas Series object, index is 73k index in [0 72999]
        '''
        if not hasattr(self.nsda, 'stim_descriptions'): # nsda only loads stim_description file to memory once, if this has not happened yet do it now
                self.nsda.stim_descriptions = pd.read_csv(
                    self.nsda.stimuli_description_file, index_col=0)
        stim_descr = self.nsda.stim_descriptions
        # get  image ids
        if shared:
            coco_ids = stim_descr[stim_descr['shared1000']==True]['cocoId']
        else:
            coco_ids = stim_descr['cocoId']
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

    def trialID_to_sess_trial(self, data_info, subj, trialID): # TODO untested
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
    
    def create_image_split(self, test_fraction=.2, shared=True):
        '''
        randomly splits all trials into a training and a test set so that each stimulus only occurs in one set
        
        INPUTS:
            test_fraction   - fraction of stimuli to hold out for testing
            shared          - if True only include stimuli seen by all participants
        '''
        ids = self.get_stimulus_ids(shared)
        num_ids = len(ids)
        num_test = ceil(num_ids * test_fraction)
        num_train = num_ids - num_test
        
        ids = ids.sample(frac=1, replace=False) # randomly sampling all datapoints in pandas is equivalent to shuffle
        train_stimuli = ids.iloc[0:num_train]
        test_stimuli = ids.iloc[num_train:]

        return train_stimuli, test_stimuli

    def trials_for_stim(self, subjects, id_frame):
        '''
        returns a list of all trials for a given participant in which the given stimuli were shown

        INPUTS:
            subjects - list of subjects for which to retrieve trial data
            id_frame - pandas frame with stimulus indices as produced by create_imgage_split
        '''
        # initialize dataframe to return
        trial_info = pd.DataFrame()
        for subj in subjects:
            # get subject information
            behaviour = pd.read_csv(self.nsda.behavior_file.format(
            subject=subj), delimiter='\t')
            print(f"NUM UNIQUE IDS: {len(behaviour['73KID'].unique())}")
            print(F"NUM UNIQUE SELECTOR TRIALS: {len(id_frame['ID73K'].unique())}")
            # select stimuli by inner join on 73K index
            stim_behav = behaviour[behaviour['73KID'].isin(id_frame['ID73K'].to_list())]
            print(f"SELECTED TRIALS: {len(stim_behav)}")
            stim_behav.assign(SUBJECT=subj)
            trial_info = trial_info.append(stim_behav)
        return trial_info
        
        


# test cases

if __name__ == "__main__":
    nsdl = NSDLoader(ROOT)
    train_stimuli, test_stimuli = nsdl.create_image_split(shared=True)
    trialdata = nsdl.trials_for_stim(['subj01'], train_stimuli)
    trialdata_test = nsdl.trials_for_stim(['subj01'], test_stimuli)

    
    '''
    # print summary of available data
    print("DATA SUMMARY")
    nsdl.get_data_info(verbose=True)
    print("#####\n")
    # test for one datapoint
    SESSION = 1
    SUBJECT = "subj01"
    captions, betas, im = nsdl.get_info_by_trial(SUBJECT, SESSION, [42])

    # shared stimuli cocoIDs
    print(nsdl.shared_imgs())

    print(captions)
    print(betas.shape)
    plt.figure(figsize=(12,4))
    plt.imshow(im)
    plt.show()
    '''
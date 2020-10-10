'''
ADAPTED FROM https://github.com/tknapen/nsd_access (check for more complete set of options)
'''

import os.path as op
import numpy as np
import nibabel as nb
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from os import listdir
import regex as re

'''
PATHS ON NAS:

preprocessed nsd:   $ROOT/nsddata/ppdata
betas:              $ROOT/nsddata_betas/ppdata
stimuli:            $ROOT/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
'''

# LOCAL PATHS, data structure was kept the same as on the NAS, but only a small subset of the data is needed
ROOT = op.join("..", "nsd") # root of the data folder, relative to this script. should probably be absolute in final script
BETAS = op.join(ROOT, "nsddata_betas", "ppdata") # different formats of betas
BEHAVIOR = op.join(ROOT, 'nsddata', 'ppdata', '{subject}', 'behav', 'responses.tsv') # information about trials, needed to get 73k index of stimuli
STIMULI = op.join(ROOT, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5') # all visual stimuli used in the experiment (no easy way to subset since all stimuli live in one hdf5 file (~37Gb))
STIM_DESCR = pd.read_csv(op.join(ROOT, 'nsddata','experiments','nsd','nsd_stim_info_merged.csv'), index_col=0) # info about which COCO subset stimuli come from
ANNOT_FILE = op.join(
            ROOT, 'nsddata_stimuli', 'stimuli', 'nsd', 'annotations', '{}_{}.json') # COCO annotations, train and validation split. maybe these can be merged into one file for convenience? (check for overlapping 73k indices)



def read_betas(subject, session_index, trial_index=[], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage'): # TODO: from the data documentation it seems that betas were shifted for storage efficiency, shift them back here?
    """read_betas read betas from MRI files

    Parameters
    ----------
    subject : str
        subject identifier, such as 'subj01'
    session_index : int
        which session, counting from 1
    trial_index : list, optional
        which trials from this session's file to return, by default [], which returns all trials
    data_type : str, optional
        which type of beta values to return from ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR', 'restingbetas_fithrf'], by default 'betas_fithrf_GLMdenoise_RR'
    data_format : str, optional
        what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'
    mask : numpy.ndarray, if defined, selects 'mat' data_format, needs volumetric data_format
        binary/boolean mask into mat file beta data format.

    Returns
    -------
    numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
        the requested per-trial beta values
    """
    data_folder = op.join(BETAS,
                          subject, data_format, data_type)
    si_str = str(session_index).zfill(2)

    if data_format == 'fsaverage':
        session_betas = []
        for hemi in ['lh', 'rh']:
            hdata = nb.load(op.join(
                data_folder, f'{hemi}.betas_session{si_str}.mgh')).get_data()
            session_betas.append(hdata)
        out_data = np.squeeze(np.vstack(session_betas))
    else:
        # if no mask was specified, we'll use the nifti image
        out_data = nb.load(
            op.join(data_folder, f'betas_session{si_str}.nii.gz')).get_data()

    if len(trial_index) == 0:
        trial_index = slice(0, out_data.shape[-1])

    return out_data[..., trial_index]


def read_behavior(subject, session_index, trial_index=[]):
        """read_behavior [summary]

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        session_index : int
            which session, counting from 0
        trial_index : list, optional
            which trials from this session's behavior to return, by default [], which returns all trials

        Returns
        -------
        pandas DataFrame
            DataFrame containing the behavioral information for the requested trials
        """

        behavior = pd.read_csv(BEHAVIOR.format(
            subject=subject), delimiter='\t')

        # the behavior is encoded per run.
        # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
        # i.e. using indexing per session, and not using the 'run' information.
        session_behavior = behavior[behavior['SESSION'] == session_index]

        if len(trial_index) == 0:
            trial_index = slice(0, len(session_behavior))

        return session_behavior.iloc[trial_index]

def read_images(image_index, show=False):
        """read_images reads a list of images, and returns their data

        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return
        show : bool, optional
            whether to also show the images, by default False

        Returns
        -------
        numpy.ndarray, 3D
            RGB image data
        """

        sf = h5py.File(STIMULI, 'r')
        sdataset = sf.get('imgBrick')
        if show:
            f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6*len(image_index), 6))
            if len(image_index) == 1:
                ss = [ss]
            for s, d in zip(ss, sdataset[image_index]):
                s.axis('off')
                s.imshow(d)
        return sdataset[image_index]

def read_image_coco_info(image_index, info_type='captions', show_annot=False, show_img=False):
        """image_coco_info returns the coco annotations of a single image or a list of images

        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return the captions for
        info_type : str, optional
            what type of annotation to return, from ['captions', 'person_keypoints', 'instances'], by default 'captions'
        show_annot : bool, optional
            whether to show the annotation, by default False
        show_img : bool, optional
            whether to show the image (from the nsd formatted data), by default False

        Returns
        -------
        coco Annotation
            coco annotation, to be used in subsequent analysis steps

                Example
                -------
                single image:
                        ci = read_image_coco_info(
                            [569], info_type='captions', show_annot=False, show_img=False)
                list of images:
                        ci = read_image_coco_info(
                            [569, 2569], info_type='captions')

        """
        
        if len(image_index) == 1:
            subj_info = STIM_DESCR.iloc[image_index[0]]

            # checking whether annotation file for this trial exists.
            # This may not be the right place to call the download, and
            # re-opening the annotations for all images separately may be slowing things down
            # however images used in the experiment seem to have come from different sets.
            annot_file = ANNOT_FILE.format(
                info_type, subj_info['cocoSplit'])
            
            coco = COCO(annot_file)
            coco_annot_IDs = coco.getAnnIds([subj_info['cocoId']])
            coco_annot = coco.loadAnns(coco_annot_IDs)

            if show_img:
                read_images(image_index, show=True)

            if show_annot:
                # still need to convert the annotations (especially person_keypoints and instances) to the right reference frame,
                # because the images were cropped. See image information per image to do this.
                coco.showAnns(coco_annot)

        elif len(image_index) > 1:

            # we output a list of annots
            coco_annot = []

            # load train_2017
            annot_file = ANNOT_FILE.format(
                info_type, 'train2017')
            coco_train = COCO(annot_file)

            # also load the val 2017
            annot_file = ANNOT_FILE.format(
                info_type, 'val2017')
            coco_val = COCO(annot_file)

            for image in image_index:
                subj_info = STIM_DESCR.iloc[image]
                if subj_info['cocoSplit'] == 'train2017':
                    coco_annot_IDs = coco_train.getAnnIds(
                        [subj_info['cocoId']])
                    coco_ann = coco_train.loadAnns(coco_annot_IDs)
                    coco_annot.append(coco_ann)

                elif subj_info['cocoSplit'] == 'val2017':
                    coco_annot_IDs = coco_val.getAnnIds(
                        [subj_info['cocoId']])
                    coco_ann = coco_val.loadAnns(coco_annot_IDs)
                    coco_annot.append(coco_ann)

        return coco_annot

def get_info_by_trial(subj, sess, trial):
    '''
    PARAMS:
        subj: subject identifier string e.g. 'subj01'
        sess: session number, int (starting to count from 1)
        trial: trial number to get
    
    RETURNS:
        tuple (stimulus, captions, betas)
        with
            stimulus: image presented at trial
            captions: list of captions from COCO
            betas: beta values for trial
    '''
    behav_data = read_behavior(SUBJECT, SESSION, trial_index=[TRIAL_IDX])
    im_id_73_info = behav_data['73KID']
    idx_73k = im_id_73_info[TRIAL_IDX]
    betas = read_betas(SUBJECT, SESSION, trial_index=[TRIAL_IDX], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage')
    captions = [x['caption'] for x in read_image_coco_info([idx_73k])]
    im = read_images([idx_73k], show=False)

    return im, captions, betas

def get_data_info(verbose=False):
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
    beta_dir = os.listdir(BETAS)
    r = re.compile("subj\d\d") # subject folders have shape subjxx, with xx two digits e.g. subj01
    subjects = [x for x in beta_dir if r.match(x)] # list of all subjects for which we have betas locally
    data_per_subj = dict()
    for s in subjects:
        
        if verbose:
            print(s)
        
        behavior = pd.read_csv(BEHAVIOR.format(subject=s), delimiter='\t') # open info file for subject
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

def shared_imgs(stim_descr):
    '''
    INPUT:
        pandas object of stimulus description file
    
    OUTPUT:
        cocoIDs of all stimuli that were shown to all participants, pandas Series object
    '''
    # get shared image ids
    coco_id_shared = stim_descr[stim_descr['shared1000']==True]['cocoId']
    return coco_id_shared

def get_stim_reps_per_subj(stim_descr, subj):
    '''
    INPUT:
        stim_descr - pandas data frame of stimulus description file
        subj - integer value 1..8 indicating the subject for which to get data
    
    RETURNS:
        rep_frame: two column pandas dataframe with '73K' the index of the stimulus, 'n_reps' number of repetitions for this subject
                   only contains entries where n_reps > 0
    '''
    # filter for columns relevant to subject, starting from column 16 each subject has 3 columns containing either 0 or the trial id of the repretition
    stim_data_subj = stim_descr.iloc[:,16+(subj-1)*3:16+(subj-1)*3+3]
    stim_reps_subj = stim_data_subj.gt(0).sum(axis=1) # finds all values greater than zero and sums over columns
    stim_reps_subj = stim_reps_subj[stim_reps_subj.gt(0)] # filter for images that were shown to the subject
    stim_reps_73k = stim_reps_subj.index.to_series() # get 73k indices of images
    rep_frame = pd.concat([stim_reps_73k, stim_reps_subj], axis=1)
    rep_frame.columns = ['73K', 'n_reps']
    return rep_frame

def trialID_to_sess_trial(data_info, subj, trialID): # TODO TEST!
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

#%% Test Code

print("TESTING...")
SUBJECT = "subj01"
TRIAL_IDX = 200
SESSION = 1

# print summary of data
data_info = get_data_info(verbose=True)


image, captions, betas = get_info_by_trial(SUBJECT, SESSION, TRIAL_IDX)

plt.figure()
plt.imshow(np.squeeze(image))
for c in captions:
    print(c)
print(f"beta info: {type(betas)}, size: {betas.shape}")
plt.show()

shared_coco = shared_imgs(STIM_DESCR)
get_stim_reps_per_subj(STIM_DESCR, 1)


#%% explore

subj = 1
stim_descr = STIM_DESCR
stim_data_subj = stim_descr.iloc[:,16+(subj-1)*3:16+(subj-1)*3+3]
stim_data_subj

mask1 = stim_data_subj['subject1_rep1'] > 0
mask2 = stim_data_subj['subject1_rep1'] < 750
firstsessrep = stim_data_subj[mask1 & mask2]

firstsessrep


beta_trial_217_486 = read_betas(SUBJECT, SESSION, trial_index=[216, 485]) # note trial indices start counting from 1
beta_trial_217_486
(beta_trial_217_486[:,0] == beta_trial_217_486[:,1]).all() # betas for multiple presentations of same stimulus are not the same
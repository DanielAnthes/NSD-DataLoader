'''
utility for creating tfrecords datasets
'''


import os
import numpy as np
import tensorflow as tf
import h5py
import re
import matplotlib.image as img
from PIL import Image, ImageOps


def rgb2gray(image):
    '''
    helper function for converting rgb image to grayscale
    '''
    gray_im = np.dot(image, np.array([0.2126, 0.7152, 0.0722]))
    return gray_im


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_record(data, subject):
    '''
    creates a tensorflow example from one datapoint of the NSD dataset

    data    -   beta values for a single trial
    subject -   subject the betas belong to
    '''
    dim = data.shape[0]
    data = tf.convert_to_tensor(data, dtype=tf.float32)  # convert to tensorflow datatype
    feature = {
        'dimension': _int64_feature(dim),
        'subject': _int64_feature(subject),
        'betas': _bytes_feature(tf.io.serialize_tensor(data))
    }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example

def create_image_record(data, subject):
    '''
    creates a tensorflow example from one datapoint of the NSD dataset

    data    -   beta values for a single trial
    subject -   subject the betas belong to
    '''
    xdim = data.shape[0]
    ydim = data.shape[1]
    data = tf.convert_to_tensor(data, dtype=tf.float32)  # convert to tensorflow datatype
    feature = {
        'xdim': _int64_feature(xdim),
        'ydim': _int64_feature(ydim),
        'subject': _int64_feature(subject),
        'betas': _bytes_feature(tf.io.serialize_tensor(data))
    }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example



def read_tfrecord(serialized_example):
    '''
    parses a serialized record
    '''
    feature_description = {
        'dimension': tf.io.FixedLenFeature((), tf.int64),
        'subject': tf.io.FixedLenFeature((), tf.int64),
        'betas': tf.io.FixedLenFeature((), tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    dimension = example['dimension']
    subject = example['subject']
    betas = tf.io.parse_tensor(example['betas'], out_type=tf.float32)
    return dimension, subject, betas


def read_image_tfrecord(serialized_example):
    '''
    parses a serialized record
    '''
    feature_description = {
        'xdim': tf.io.FixedLenFeature((), tf.int64),
        'ydim': tf.io.FixedLenFeature((), tf.int64),
        'subject': tf.io.FixedLenFeature((), tf.int64),
        'betas': tf.io.FixedLenFeature((), tf.string)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    dimension = (example['xdim'], example['ydim'])
    subject = example['subject']
    betas = tf.io.parse_tensor(example['betas'], out_type=tf.float32)
    return dimension, subject, betas


def write_dataset_to_tfrecords(data, subject, prefix, shard_size=500):
    '''
    write dataset to disk as multiple tfrecord files

    data        -   data as a hdf5 object
    prefix      -   filename
    shard_size  -   number of datapoints to write to each tfrecords file (max)
    '''
    n_data = data.shape[0]
    i = 0
    shard_num = 0
    while i < n_data:
        if i + shard_size < n_data:
            d = data[i:i+shard_size]
        else:
            d = data[i:]
        
        filename = f"{prefix}_{shard_num}.tfrecords"
        with tf.io.TFRecordWriter(filename) as writer:
            for dp in d:
                example = create_record(dp, subject)
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)

        i += shard_size
        shard_num += 1


def create_record_with_info(betas, subj, sess, id73k, sess_idx):
    '''
    creates a tensorflow example from one datapoint of the NSD dataset
    includes information about data to uniquely identify datapoint

    betas    -   beta values for a single trial
    subj     -  subject number
    sess     -  recording session from NSD info
    id73k    -  stimulus id from NSD dataset
    sess_idx -  index within session form NSD recording
    '''
    dim = betas.shape[0]
    data = tf.convert_to_tensor(betas, dtype=tf.float32)  # convert to tensorflow datatype
    feature = {
        'dimension': _int64_feature(dim),
        'subject': _int64_feature(subj),
        'betas': _bytes_feature(tf.io.serialize_tensor(data)),
        'sess' : _int64_feature(sess),
        'id73k': _int64_feature(id73k),
        'sess_idx': _int64_feature(sess_idx)
    }
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    return example


    

def write_batch_to_tfrecord(betas, info ,filename):
    '''
    takes a single batch of data from memory and writes it to
    a single tfrecords file

    data        -   datapoints to be written to tfrecords file
    filename    -   path of tfrecords file
    '''
    with tf.io.TFRecordWriter(filename) as writer:
        for b, i in zip(betas, info.to_numpy()):
            subject, session, id73k, sess_idx = i[0], i[1], i[4], i[5]
            example = create_record_with_info(b, subject, session, id73k, sess_idx)
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)



def images_to_tfrecords(files, subject, prefix, shard_size=500):
    '''
    takes a list of images and saves all images as a tfrecords dataset,
    groups images together in multiple separate tfrecords files as given by
    shard_size parameter
    '''
    n_data = len(files)
    i = 0
    shard_num = 0
    while i < n_data:
        print(f"writing shard {shard_num}", end="\r")
        if i + shard_size < n_data:
            d = files[i:i+shard_size]
        else:
            d = files[i:]

        filename = f"{prefix}_{shard_num}.tfrecords"
        with tf.io.TFRecordWriter(filename) as writer:
            for numf in enumerate(d):
                print(f"writing image {numf}", end="\r")
                im = img.imread(f)
                im = im[:,:,:3]  # remove alpha channel
                im = rgb2gray(im)
                example = create_image_record(im, subject)
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)

        i += shard_size
        shard_num += 1


def read_tfrecords(dataset_name, working_dir='.', subset=[], num_parallel=1):
    '''
    takes the name (prefix) for a tfrecords dataset with one or more files
    and returns a TFRecordDataset instance

    alternatively, specifying a number of shards explicitly will create a dataset from the specified shards
    '''

    if len(subset) == 0:
        files = os.listdir(working_dir)
        dataset_re = re.compile(f"{dataset_name}_\d+.tfrecords")
        data_files = dataset_re.findall(''.join(files))
    
    else:
        data_files = [f"{dataset_name}_{i}.tfrecords" for i in subset]

    print(f"selected {len(data_files)} files belonging to dataset {dataset_name}")
    
    data_paths = [os.path.join(working_dir, file) for file in data_files]
    tfrecord_dataset = tf.data.TFRecordDataset(data_paths, num_parallel_reads=num_parallel)
    return tfrecord_dataset


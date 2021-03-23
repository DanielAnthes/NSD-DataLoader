import nsdtf
import os

NSDDIR = '/home/daniel/Documents/Masterarbeit/NSD'
TYPE = 'betas_fithrf_GLMdenoise_RR'
FORMAT = 'fsaverage'

loader = nsdtf.BufferedLoader(NSDDIR, TYPE, FORMAT, ["subj02"], 4)

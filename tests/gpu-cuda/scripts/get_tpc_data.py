#/usr/bin/env python

import numpy as np
from pathlib import Path
import sys


layer_group = sys.argv[1]
num_examples = int(sys.argv[2])
save_folder = '../data/'

tpc_data_path = f'/hpcgpfs01/scratch/yhuang2/TPC/highest_framedata_3d/{layer_group}'

fnames = [path for path in Path(tpc_data_path).rglob('*.npy')]

sampled_fnames = np.random.choice(fnames, num_examples, replace=False)
for i, fname in enumerate(sampled_fnames):
	tokens = str(fname.name).split('.')
	id1 = tokens[0].split('_')[-1]
	id2 = tokens[1].split('_')[-1]
	
	data = np.load(fname)
	save_fname = f'../data/tpc_{layer_group}_{id1}_{id2}.dat'
	data.tofile(save_fname)

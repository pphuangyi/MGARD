#/usr/bin/env python

import numpy as np
from pathlib import Path
import sys


layer_group = sys.argv[1]
num_examples = int(sys.argv[2])
data_type = sys.argv[3]
assert data_type in ['uint16', 'float32', 'float64'], \
	print("choose data type in ['uint16', 'float32', 'float64']")

save_folder = '../tpc_data/'


tpc_data_path = f'/hpcgpfs01/scratch/yhuang2/TPC/highest_framedata_3d/{layer_group}'

fnames = [path for path in Path(tpc_data_path).rglob('*.npy')]

sampled_fnames = np.random.choice(fnames, num_examples, replace=False)
for i, fname in enumerate(sampled_fnames):
	tokens = str(fname.name).split('.')
	id1 = tokens[0].split('_')[-1]
	id2 = tokens[1].split('_')[-1]

	data = np.load(fname)
	
	if data_type == 'uint16':
		save_fname = f'{save_folder}/tpc_{layer_group}_{id1}_{id2}.dat'
	elif data_type == 'float32':
		data = data.astype(np.float32)
		save_fname = f'{save_folder}/tpc_{layer_group}_{id1}_{id2}_float.dat'
	else:
		data = data.astype(np.float64)
		save_fname = f'{save_folder}/tpc_{layer_group}_{id1}_{id2}_double.dat'

	data.tofile(save_fname)

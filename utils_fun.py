import numpy as np


def split_question(ids, mask, segment,y):
	new_ids = []; new_mask = []; new_segment = []; new_label=[]

	padding = [0] * (ids.shape[1] - 64)	
	for each_id, each_mask, each_segment, label in zip(ids, mask, segment, y):
		valid_len = len(np.where(each_mask==1)[0])
		for idx in range(0,valid_len,64):
			#filter content longer than max sequence length, ex.512
			if idx+64>len(each_id):
				continue
			if len(new_label) >= 32:
				break		
	
			new_ids.append(each_id[idx:idx+64].tolist() + padding)
			new_mask.append(each_mask[idx:idx+64].tolist() + padding)
			new_segment.append(each_segment[idx:idx+64].tolist() + padding)
			new_label.append(label)

	return new_ids, new_mask, new_segment, new_label

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
import tokenization


def convert_single_example(question, max_seq_length, tokenizer):
	text_a = tokenization.convert_to_unicode(question)
	tokens_a = tokenizer.tokenize(text_a)
	if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)

	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	#print("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
	#print("input_ids: %s" % " ".join([str(x) for x in input_ids[:50]]))
	#print("input_mask: %s" % " ".join([str(x) for x in input_mask[:50]]))
	#print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

	return input_ids, input_mask, segment_ids

def get_type_dict():
	type_dict = {}	
	with open("data/label_stat.txt",'r') as f2:
		txt = f2.readlines()
	for i, line in enumerate(txt):
		l = line.strip().split()
		type_name =l[1]
		
		#type_dict[type_name] = str(i)
		type_dict[i] = type_name
	return type_dict


label_dict = get_type_dict()



vocab_file = "model_files/chinese_L-12_H-768_A-12/vocab.txt"
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

pb_file = 'output/v2_testB_model.pb'
#pb_file = '../003_test_module/model_files/v2.2_testB_model.pb'
with tf.gfile.FastGFile(pb_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=g_in)




def inference(question):
	max_seq_length = 512
	input_ids, input_mask, segment_ids = convert_single_example(
				question, max_seq_length, tokenizer)
	#output_node_names=["input_ids", "input_mask", "logits", "segment_ids"])
	x = sess.graph.get_tensor_by_name("input_ids:0")
	mask = sess.graph.get_tensor_by_name("input_mask:0")
	segment = sess.graph.get_tensor_by_name("segment_ids:0")
	logits = sess.graph.get_tensor_by_name("logits:0")

	z = sess.run(logits, feed_dict={
									x:[input_ids], mask: [input_mask], 
									segment: [segment_ids]
								})
	z = z[0]
	z[z>10] = 10
	z[z<-10] = -10
	y_ = 1/(1+np.exp(-z))

	idx = np.where(y_>0.75)[0]
	#idx = np.argmax(y_)
	prob = y_[idx]
	label = [label_dict[x] for x in idx]
	#print(label_dict[idx])
	pred_dict = {}
	for y,p in zip(label,prob):
		pred_dict[y] = p
	out_info = ''
 
	sq_list = []
	prob_list = []
	for y,p in sorted(pred_dict.items(), key=lambda x:x[1], reverse=True):
		buf = y.split('-')
		assert len(buf) == 2
		jf, sq = buf
		if len(sq_list) == 0:
			main_jf = jf	
		if jf != main_jf:
			continue
	
		sq_list.append(sq)
		prob_list.append(p)
		out_info += y + ',' + str(round(p,3)) + '; '
	if len(sq_list) == 0:
		return '',0,0,0 
	else:
		return ','.join(sq_list), [main_jf + '-' +x for x in sq_list] ,main_jf, round(np.mean(prob_list),3)
	#for y,p in sorted(pred_dict.items(), key=lambda x:x[1], reverse=True):
	#	out_info += y + ',' + str(round(p,3)) + '; '
	#return out_info.strip('; '), label


if __name__ == "__main__":
	#filename = "data/qa_testset_B.txt"
	#filename = "data/qa_testset_C.txt"
	filename = "data/labelled_qa_testset_B.txt"
	with open(filename,'r') as f1:
		txt = f1.readlines()

	#question = "合同没到期，公司要裁员，如何赔偿"
	correct = 0
	FP = 0
	total = 0
	total2 = 0
	cate_acc = {}
	for line in txt:
		l = line.strip().split('\t')
		question = l[1]
		y = l[2]
		y_, pred, jf, prob = inference(question)
		if prob == 0:
			continue	

		y2 = y.split(',')
		gt_jf = y2[0].split('-')[0]
		gt_sq = ','.join([x.split('-')[1] for x in y2])

		total += 1
		total2 += 1
		TP_num = len([x for x in y.split(',') if x in pred])
		FP_num = len([x for x in pred if x in y])
		if TP_num == len(y.split(',')):
			correct += 1
		elif TP_num != 0:
			correct += 0.5
		if FP_num == len(pred):
			FP += 1
		elif FP_num != 0:
			FP += 0.5

		print(question + '\n' + 'prediction: ' + jf + '\t' +  y_ + '\n' + 'label: ' + gt_jf + '\t' + gt_sq +  '\n')
	print("recall: %f" % (float(correct)/total))
	print("precision: %f" % (float(FP)/total2))





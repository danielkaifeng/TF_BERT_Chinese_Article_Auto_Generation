from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import modeling
import optimization
import tokenization
import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file", None,"This specifies the model architecture.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", "./output", "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("init_checkpoint", "model_files/chinese_L-12_H-768_A-12/bert_model.ckpt", "from a pre-trained BERT model).")
flags.DEFINE_bool( "do_lower_case", True, " Should be True for uncased models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 128, "than this will be padded.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool( "do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 1.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")


class InputExample(object):
	def __init__(self, guid, text_a, text_b=None, label=None):
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class PaddingInputExample(object):
	"""Fake example so the num input examples is a multiple of the batch size.
	"""


class InputFeatures(object):
	def __init__(self,
							 input_ids,
							 input_mask,
							 segment_ids,
							 label_id,
							 is_real_example=True):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.is_real_example = is_real_example


class DataProcessor(object):
	def get_train_examples(self, data_dir):
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		raise NotImplementedError()

	def get_test_examples(self, data_dir):
		raise NotImplementedError()

	def get_labels(self):
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		with tf.gfile.Open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines



class MrpcProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_test_examples(self, data_dir):
		return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

	def get_labels(self):
		with open("label_stat.txt",'r') as f2:
				txt = f2.readlines()
		label_list = [x.strip().split()[1] for x in txt]
		return label_list

	def _create_examples(self, lines, set_type):
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[1])
			text_b = tokenization.convert_to_unicode(line[2])
			if set_type == "test":
				label = "0"
			else:
				label = tokenization.convert_to_unicode(line[0])
			examples.append(
					InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples



def convert_single_example(ex_index, example, label_list, max_seq_length,
													 tokenizer):
	if isinstance(example, PaddingInputExample):
		return InputFeatures(
				input_ids=[0] * max_seq_length,
				input_mask=[0] * max_seq_length,
				segment_ids=[0] * max_seq_length,
				label_id=0,
				is_real_example=False)

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)

	if tokens_b:
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
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

	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	label_id = label_map[example.label]
	if ex_index < 2:
		tf.logging.info("*** Example ***")
		tf.logging.info("guid: %s" % (example.guid))
		tf.logging.info("tokens: %s" % " ".join(
				[tokenization.printable_text(x) for x in tokens]))
		tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
		tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

	feature = InputFeatures(
			input_ids=input_ids,
			input_mask=input_mask,
			segment_ids=segment_ids,
			label_id=label_id,
			is_real_example=True)
	return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
	writer = tf.python_io.TFRecordWriter(output_file)

	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		feature = convert_single_example(ex_index, example, label_list,
																		 max_seq_length, tokenizer)

		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f

		features = collections.OrderedDict()
		features["input_ids"] = create_int_feature(feature.input_ids)
		features["input_mask"] = create_int_feature(feature.input_mask)
		features["segment_ids"] = create_int_feature(feature.segment_ids)
		features["label_ids"] = create_int_feature([feature.label_id])
		features["is_real_example"] = create_int_feature(
				[int(feature.is_real_example)])

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())
	writer.close()


def get_input_data(input_file, seq_length, batch_size):
	def parser(record):
			name_to_features = {
					"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
					"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
					"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
					"label_ids": tf.FixedLenFeature([], tf.int64),
					#"is_real_example": tf.FixedLenFeature([], tf.int64),
			}

			example = tf.parse_single_example(record, features=name_to_features)
			input_ids = example["input_ids"]
			input_mask = example["input_mask"]
			segment_ids = example["segment_ids"]
			labels = example["label_ids"]
			return input_ids, input_mask, segment_ids, labels
	
	dataset = tf.data.TFRecordDataset(input_file)
	dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=1000)
	iterator = dataset.make_one_shot_iterator()
	input_ids, input_mask, segment_ids, labels = iterator.get_next()
	
	return input_ids, input_mask, segment_ids, labels


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
								 labels, num_labels, use_one_hot_embeddings):
	model = modeling.BertModel(
			config=bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=use_one_hot_embeddings)

	output_layer = model.get_pooled_output()

	hidden_size = output_layer.shape[-1].value

	output_weights = tf.get_variable(
			"output_weights", [num_labels, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

	output_bias = tf.get_variable(
			"output_bias", [num_labels], initializer=tf.zeros_initializer())

	with tf.variable_scope("loss"):
		if is_training:
			# I.e., 0.1 dropout
			output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		probabilities = tf.nn.softmax(logits, axis=-1)
		log_probs = tf.nn.log_softmax(logits, axis=-1)

		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)

		return (loss, per_example_loss, logits, probabilities)

def convert_label_to_multi_hot(labels, num_classes):
		output_label = []
		for label_idx in labels:
				multi_hot = [0] * num_classes
				for i in label_idx:
						multi_hot[i] = 1
				output_label.append(multi_hot)

		return np.array(output_label)

def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)
	tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

	if FLAGS.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError(
				"Cannot use sequence length %d because the BERT model "
				"was only trained up to sequence length %d" %
				(FLAGS.max_seq_length, bert_config.max_position_embeddings))

	tf.gfile.MakeDirs(FLAGS.output_dir)

	processor = MrpcProcessor()
	label_list = processor.get_labels()

	tokenizer = tokenization.FullTokenizer(
			vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	if 0:
		train_examples = processor.get_train_examples(FLAGS.data_dir)
		train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
		file_based_convert_examples_to_features(
			train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

		eval_examples = processor.get_dev_examples(FLAGS.data_dir)
		eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
		file_based_convert_examples_to_features(
			eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

	train_examples = None
	num_train_steps = None
	num_warmup_steps = None
	train_examples = processor.get_train_examples(FLAGS.data_dir)
	num_train_steps = int(
			len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
	num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


	is_training = True
	
	seq_len = FLAGS.max_seq_length
	input_ids = tf.placeholder(tf.int64, shape=[None, seq_len])
	input_mask = tf.placeholder(tf.int64, shape=[None, seq_len])
	segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len])
	labels = tf.placeholder(tf.int64, shape=[None])
	num_labels = len(label_list)
	use_one_hot_embeddings = False

	loss, per_example_loss, logits, probabilities = create_model(
									bert_config, is_training, input_ids, input_mask, segment_ids,
									labels, num_labels, use_one_hot_embeddings)

	input_file = "output/train.tf_record"
	batch_size = FLAGS.train_batch_size
	input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(input_file, seq_len, batch_size)
	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

	val_input_file = "output/eval.tf_record"
	val_batch_size = FLAGS.eval_batch_size
	val_input_ids2, val_input_mask2, val_segment_ids2, val_labels2 = get_input_data(val_input_file, seq_len, val_batch_size)
	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

	init_global = tf.global_variables_initializer()
	saver=tf.train.Saver()


	with tf.Session() as sess:
		sess.run(init_global)
		lastest_checkpoint = tf.train.latest_checkpoint('output')
		saver.restore(sess, lastest_checkpoint)

		for i in range(200000):
			ids, mask, segment,y = sess.run([input_ids2, input_mask2, segment_ids2, labels2])
			feed = {input_ids:ids, input_mask: mask, segment_ids: segment, labels:y}
			_, out_loss, out_logits = sess.run([optimizer, loss,logits], feed_dict=feed)
	
			if i % 20 == 0:	
				ids, mask, segment,y = sess.run([val_input_ids2, val_input_mask2, val_segment_ids2, val_labels2])
				feed = {input_ids:ids, input_mask: mask, segment_ids: segment, labels:y}
				val_loss, prob = sess.run([loss,probabilities], feed_dict=feed)
				print(y)
				val_acc = np.mean(np.int16(np.argmax(prob,1) == y))

			log_info = "step %d - loss: %f val_loss: %f val_acc: %f" % (i, out_loss, val_loss, val_acc)
			print(log_info)
		if i % 200 == 0:	
			saver.save(sess, 'output/bert_v1.ckpt')


if __name__ == "__main__":
	flags.mark_flag_as_required("data_dir")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("bert_config_file")
	tf.app.run()

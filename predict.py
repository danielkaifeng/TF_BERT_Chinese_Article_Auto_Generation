import tensorflow as tf



pb_file = 'output/model.pb'
with tf.gfile.FastGFile(pb_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=g_in)


def inference(x):
	x = sess.graph.get_tensor_by_name("input_ids:0")



	z = sess.run(logits, feed_dict={})

	return y_


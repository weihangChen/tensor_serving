from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf
import numpy as np
import os
import shutil

def main(_):
    with tf.name_scope('name_scope_x'):
        input = tf.placeholder(dtype= tf.int32,shape= [None, 4], name='input')
        b = tf.constant([1,1,1,1], dtype=tf.int32, shape=[4, 1])
        output = tf.matmul(input,b, name='output')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    #save model
    export_dir = 'D:/data/1'
    try:
        shutil.rmtree(export_dir)
    except OSError:
        pass
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    signature_def_map = {
                    'mypredict': signature_def_utils.predict_signature_def(inputs= {"input": input}, outputs= {"output": output})
               }
    
    builder.add_meta_graph_and_variables(sess,
      tags = [tag_constants.SERVING],
      signature_def_map = signature_def_map,
      assets_collection = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      legacy_init_op = None,
      clear_devices = True)
    builder.save()
    #restore, only if this part works, then you should start the implementation
    #for grpc client, else there is no way to understand the exception tf.reset_default_graph()
    
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name("name_scope_x/input:0")
        output_tensor = graph.get_tensor_by_name("name_scope_x/output:0")
        tmp = sess.run(output_tensor, feed_dict={input_tensor: [[1,2,3,4]] })
        #this should yield [10]
        print(tmp)
    print('done')
    

if __name__ == '__main__':
    tf.app.run(main=main)
    

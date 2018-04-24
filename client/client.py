from __future__ import print_function

import sys
import threading
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.saved_model import tag_constants

tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport):
  try:
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    request = predict_pb2.PredictRequest()
    #should be same name as in the 
    #ENTRYPOINT [ "tensorflow_model_server", "--port=9000", "--model_name=mymodel", "--model_base_path=/server/trained" ]
    request.model_spec.name = 'mymodel'
    request.model_spec.signature_name = 'mypredict'
    
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(np.array([1,2,3,4], dtype=np.int32), shape=[1, 4]))
    result = stub.Predict(request, 5.0)  # 5 se.cs timeout 
    print(result)

  except Exception  as e:
    print(e)


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  do_inference(FLAGS.server)

  print('--------------done------------')

if __name__ == '__main__':
  tf.app.run()

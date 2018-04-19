from class_activation_map import *
from lenet_slim import le_net
from utils import *
#from tensorflow.python.tools import inspect_checkpoint as chkp
#following are saved in the checkpoint file           
#+ ['LeNet/conv1/weights'] [5, 5, 3, 32] list
#+ ['LeNet/conv1/biases'] [32] list
#+ ['LeNet/conv2/weights'] [5, 5, 32, 64] list
#+ ['LeNet/conv2/biases'] [64] list
#+ ['LeNet/GAP/W'] [64, 10] list
#use following to print saved variables in checkpoint            
#chkp.print_tensors_in_checkpoint_file(file_name=checkpoint, tensor_name='', all_tensors=True)
            
batch_size = 1
im_width = 100
im_height = 100
dataset_percentage = 0.1 # 1.0 takes 100k rows.  0.1 takes 10k rows.
if __name__ == '__main__':
    
    #very important to rest graph every time
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, (None, im_width, im_width, 3))
    #logits, net
    y, top_conv = le_net(images=x, num_classes=10)
    #labels
    y_ = tf.placeholder(tf.int64, [None])
    class_activation_map = get_classmap_all_labels([0,1,2,3,4,5,6,7,8,9], top_conv, im_width)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        checkpoint = "/tensor-serving/trained-model/mnist-cluttered-99900"
        try:
            step_start = saver.restore(sess, checkpoint)
        except Exception as e:
            print(e)
        print('Finished initializing the model...')
        images = read_one_image("img_19")
        labels = [4]
        

        inspect_class_activation_map(sess, class_activation_map, top_conv, images, labels, 1, 1, x, y_, y, label_w = None, show = False)
       
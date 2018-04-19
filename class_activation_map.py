import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imsave
from utils import mkdir_p
import numpy as np
from variables import *

#bebug only, print out the selected label specific weights
#def get_class_map1(label, conv, im_width):
#    output_channels = int(conv.get_shape()[-1])

#    with tf.variable_scope('LeNet/GAP', reuse=True):
#        last_layer_weights = tf.get_variable('W')
#        last_layer_weights_tranposed = tf.transpose(last_layer_weights)
#        label_w = tf.gather(last_layer_weights_tranposed, label)
#    return label_w


def get_class_map(label, conv, im_width):
    output_channels = int(conv.get_shape()[-1])

    with tf.variable_scope('LeNet/GAP', reuse=True):
        last_layer_weights = tf.get_variable('W')
        last_layer_weights_tranposed = tf.transpose(last_layer_weights)
        last_layer_weights_tranposed_trace = tf.Print(last_layer_weights_tranposed,[label, tf.shape(last_layer_weights_tranposed)],'--------GAP all weights>>', summarize = 100)
        label_w = tf.gather(last_layer_weights_tranposed_trace, label)
        label_w_trace =  tf.Print(label_w,[tf.shape(label_w), label_w],'------GAP label specific weights>>', summarize = 100)
        label_w_reshaped = tf.reshape(label_w_trace, [-1, output_channels, 1])

    #resize the last conv layer back to original image size
    conv_resized = tf.image.resize_bilinear(conv, [im_width, im_width]) 
    conv_resized_trace = tf.Print(conv_resized,[tf.shape(conv_resized)],'--------Last conv layer resized to 100 * 100>>', summarize = 100)
    conv_resized_reshaped = tf.reshape(conv_resized_trace, [-1, im_width * im_width, output_channels])
    classmap = tf.matmul(conv_resized_reshaped, label_w_reshaped)
    classmap_trace = tf.Print(classmap,[tf.shape(classmap)],'--------class map>>', summarize = 100)
    classmap_reshaped = tf.reshape(classmap_trace, [-1, im_width, im_width])
    return classmap_reshaped, last_layer_weights_tranposed, label_w



def inspect_class_activation_map(sess, class_activation_map, top_conv,
                                 images_test, labels_test, global_step,
                                 num_images, x, y_, y, label_w, show = False):
    for s in range(num_images):
        output_dir = 'images/out/img_{0}/'.format(s)
        mkdir_p(output_dir)
        imsave('{}/image_test.png'.format(output_dir), images_test[s])
        img = images_test[s:s + 1]
        label = labels_test[s:s + 1]
        conv6_val, output_val = sess.run([top_conv, y], feed_dict={x: img})
        
        classmap_answer = sess.run(class_activation_map, feed_dict={y_: label, top_conv: conv6_val})
        
        classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_answer))
        for vis, ori in zip(classmap_vis, img):
            plt.imshow(1 - ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            cmap_file = '{}/cmap_{}.png'.format(output_dir, global_step)
            plt.savefig(cmap_file)
            if show == True:
                plt.show()
            plt.close()



def get_classmap_by_label(conv, label, im_width, output_channels):
    label_w_reshaped = None
    with tf.variable_scope('LeNet/GAP', reuse=True):
        last_layer_weights = tf.get_variable('W')
        last_layer_weights_tranposed = tf.transpose(last_layer_weights)
        label_w = tf.gather(last_layer_weights_tranposed, label)
        label_w_reshaped = tf.reshape(label_w, [-1, output_channels, 1])

    classmap = tf.matmul(conv, label_w_reshaped)
    #remove all negative values https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
    classmap = tf.nn.relu(classmap)
    classmap_reshaped = tf.reshape(classmap, [-1, im_width, im_width])
    return classmap_reshaped

#this method build class activation map with all categories, not just one
def get_classmap_all_labels(labels, conv, im_width):
    output_channels = int(conv.get_shape()[-1])
    #resize the last conv layer back to original image size
    conv_resized = tf.image.resize_bilinear(conv, [im_width, im_width]) 
    conv_resized_reshaped = tf.reshape(conv_resized, [-1, im_width * im_width, output_channels])
    class_maps = [get_classmap_by_label(conv_resized_reshaped, label,im_width,output_channels) for label in labels]
    #average class_maps into one
    class_map_mean = tf.reduce_mean(class_maps, 0)
    return class_map_mean

def inspect_class_activation_map1(sess, class_activation_map, top_conv,
                                 images_test, labels_test, global_step,
                                 num_images, x, y_, y, label_w = None,  show = False):
    for s in range(num_images):
        output_dir = 'images/out/img_{1}/'.format(s)
        mkdir_p(output_dir)
        imsave('{}/image_test.png'.format(output_dir), images_test[s])
        img = images_test[s:s + 1]
        label = labels_test[s:s + 1]
        conv6_val, output_val = sess.run([top_conv, y], feed_dict={x: img})
        
        classmap_answer = sess.run(class_activation_map, feed_dict={y_: label, top_conv: conv6_val})
        
        classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_answer))
        for vis, ori in zip(classmap_vis, img):
            plt.imshow(1 - ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            cmap_file = '{}/cmap_{}.png'.format(output_dir, global_step)
            plt.savefig(cmap_file)
            if show == True:
                plt.show()
            plt.close()


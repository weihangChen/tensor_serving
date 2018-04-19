import tensorflow as tf

slim = tf.contrib.slim


def le_net(images, num_classes=10, scope='LeNet'):
    with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        c1 = slim.conv2d(images, 32, [5, 5], scope='conv1')
        p1 = slim.max_pool2d(c1, [2, 2], 2, scope='pool1')
        c2 = slim.conv2d(p1, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(c2, [2, 2], 2, scope='pool2')
        #print network structure for logging purpose
        #c1=[n, 100, 100, 32], [n, 50, 50, 32], [n,50, 50, 64],[n, 25, 25, 64]
        #net = tf.Print(p2,[tf.shape(c1),tf.shape(p1),tf.shape(c2),tf.shape(p2) ],"network structure (c1, p1, c2, p2)>>", summarize=100)
        
        #Global  average  pooling  (GAP)  vs  global  max  pool-ing (GMP), maybe should use max?
        #here is a three dimension array
        #arr = np.array([[[50,50,50],[50,50,50]],[[100,100,100],[100,100,100]]])
        #aa = np.mean(arr, axis=(1,2)) elements from dimension 1 and 2 are sum up together
        #reduce_mean produce [50,100]       

        gap = tf.reduce_mean(net, (1, 2))
        with tf.variable_scope('GAP'):
            gap_w = tf.get_variable('W', shape=[64, 10], initializer=tf.random_normal_initializer(0., 0.01))
        logits = tf.matmul(gap, gap_w)
    return logits, net


def le_net_arg_scope(weight_decay=0.0):
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc
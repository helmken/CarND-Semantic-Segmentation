import os.path
import tensorflow as tf
import helper
import time
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" 
        and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, 
        layer3_out, layer4_out, layer7_out)
    """
    
    # Use tf.saved_model.loader.load to load the model and weights
    # from the file
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    
    """
    Create the layers for a fully convolutional network. Build skip-layers 
    using the vgg layers.
    :param vgg_layer7_out: TF Tensor for output of VGG Layer 3
    :param vgg_layer4_out: TF Tensor for output of VGG Layer 4
    :param vgg_layer3_out: TF Tensor for output of VGG Layer 7
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # num_classes: binary classification: "is pixel road or not road?" 
    #     -> 2 classes
    
    # regularizer avoids weights getting too large, 
    # regularizer is important to avoid overfitting and producing garbage
    
    # instead of fully connected classifier, append 1x1 convolution to
    # vgg_layer7
    layer7_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size = 1, padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # same for vgg_layer4
    layer4_1x1 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size = 1, padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # and for vgg_layer3
    layer3_1x1 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size = 1, padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))


    # decoder part: upsample to original layer size with transpose convolutions
    vgg_layer7_trans = tf.layers.conv2d_transpose(
        layer7_1x1, num_classes, 
        kernel_size = 4, strides = (2, 2), # up-sampling 
        padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # skip connection from vgg_layer4 to up-sampled vgg_layer7
    vgg_layer4_skip = tf.add(layer4_1x1, vgg_layer7_trans)


    vgg_layer4_trans = tf.layers.conv2d_transpose(
        vgg_layer4_skip, num_classes,
        kernel_size = 4, strides = (2, 2), # up-sampling 
        padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # skip connection from vgg_layer3 to up-sampled vgg_layer4
    vgg_layer3_skip = tf.add(layer3_1x1, vgg_layer4_trans)


    vgg_layer3_trans = tf.layers.conv2d_transpose(
        vgg_layer3_skip, num_classes,
        kernel_size = 16, strides = (8, 8), # up-sampling 
        padding = 'same', 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    # debugging hint: capital P in Print is important: adds a Print node to
    # the tensorflow graph - printing is done during session run
    # tf.Print(vgg_layer7_trans, [tf.shape(vgg_layer7_trans)])
    
    return vgg_layer3_trans 
    
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    # classroom: 09 FCN-8 Classification & Loss
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = labels, logits = logits))

    train_op = tf.train.AdamOptimizer(learning_rate) \
        .minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, 
             train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data. 
            Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    for epoch in range(epochs):
        
        print('training epoch', epoch)
        t_start = time.time()
        
        for image, label in get_batches_fn(batch_size):
            
            feed_dict = {input_image : image, 
                         correct_label : label, 
                         keep_prob : 0.5}
            
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = feed_dict)
            
            #print('loss=', loss)
        
        print('epoch {}: duration: {}'.format(epoch, time.time() - t_start))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    epochs = 32
    batch_size = 1
    
    with tf.Session() as sess:
        
        #with tf.device("/gpu:0"):
        with tf.device("/cpu:0"):
        
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
        
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(
                os.path.join(data_dir, 'data_road/training'), image_shape)
            
            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
    
            # get layers from stored VGG
            image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
            
            # create graph with skip connections, return last layer, inference part
            final_layer = layers(layer3, layer4, layer7, num_classes)
    
            correct_label = tf.placeholder(
                tf.float32, 
                [None, image_shape[0], image_shape[1], num_classes])
            learning_rate = tf.constant(0.0001)
            
            # continue with graph, add loss and training part
            logits, train_op, cross_entropy_loss = optimize(
                final_layer, correct_label, learning_rate, num_classes)
    
            sess.run(tf.global_variables_initializer())
            
            print('start training')
            train_nn(sess, epochs, batch_size, get_batches_fn, 
                     train_op, cross_entropy_loss, 
                     image_input, correct_label, 
                     keep_prob, learning_rate)
    
            print('saving inference data')
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                          logits, keep_prob, image_input)
    
            # OPTIONAL: Apply the trained model to a video
            print('... finished')


if __name__ == '__main__':
    run()

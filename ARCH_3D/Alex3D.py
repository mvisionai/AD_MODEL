from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as K
from ARCH_3D import ops as op_linker

shape_to_return=None
d_W_fc0=None
d_b_fc0=None


def input_placeholder(image_size, image_channel, label_cnt,train_type="domain"):
    with tf.name_scope('inputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_size, image_channel], 'inputs')
        labels = tf.placeholder("float", [None, label_cnt], 'labels')
        training = tf.placeholder(tf.bool, [])
        dropout_keep_prob = tf.placeholder("float",None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')

        if train_type == "domain":
          domain_label = tf.placeholder("float", [None, 2], name='domain')
          return inputs, labels, training, dropout_keep_prob, learning_rate, domain_label
        elif train_type == "single":
          return inputs, labels, training, dropout_keep_prob, learning_rate


def train_inputs(image_size, image_channel, label_cnt) :
    with tf.name_scope('coninputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_size, image_channel], 'input')
        labels = tf.placeholder("float", [None, label_cnt], 'label')
        dropout_keep_prob = tf.placeholder("float", None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')
        domain_label = tf.placeholder("float", [None, 2], name='domain')
        flip_gra = tf.placeholder(tf.float32, [], name='flip')

    return inputs, labels, dropout_keep_prob, learning_rate, domain_label, flip_gra




def inference(inputs, training, dropout_keep_prob, label_cnt):
    # todo: change lrn parameters
    with tf.variable_scope("convolution"):
            with tf.variable_scope('conv1layer'):
                conv1 = op_linker.conv(inputs, 2, 32,1)   # (input data, kernel size, #output channels, stride_size)
                conv1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                conv1 = tf.layers.batch_normalization(conv1, training=training)

            # conv layer 2
            with tf.variable_scope('conv2layer'):
                conv2 = op_linker.conv(conv1,2,64, 1, 0.1)
                conv2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                conv2 = tf.layers.batch_normalization(conv2, training=training)

            # conv layer 3
            with tf.variable_scope('conv3layer'):
                conv3 = op_linker.conv(conv2, 2,128, 1, 0.1)
                conv3 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                conv3 = tf.layers.batch_normalization(conv3, training=training)
                #no bias

            # conv layer 4
            with tf.variable_scope('conv4layer'):
                conv4 = op_linker.conv(conv3,2, 256, 1, 0.1)
                conv4 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                conv4 = tf.layers.batch_normalization(conv4, training=training)

            # conv layer 5
            with tf.variable_scope('conv5layer'):
                conv5 = op_linker.conv(conv4, 2, 512, 1, 0.1)
                conv5 = tf.nn.max_pool3d(conv5, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
                conv5 = tf.layers.batch_normalization(conv5, training=training)

            # fc layer 1
            with tf.variable_scope('fc1layer'):
                global  shape_to_return
                global  d_b_fc0
                global  d_W_fc0
                fc1,shape_to_return,d_W_fc0,d_b_fc0 = op_linker.fc(conv5,512, 0.1, use_weight=True)
                fc1 = tf.nn.dropout(fc1, dropout_keep_prob)
                #bias changed

            # fc layer 2
            with tf.variable_scope('fc2layer'):
                fc2 = op_linker.fc(fc1,512, 0.1)
                fc2 = tf.nn.dropout(fc2, dropout_keep_prob)
                #bias changed

            # fc layer 3 - output
            with tf.variable_scope('fc3layer'):
                print(" ",end="\n")
                print("Graph Built")
                final_layer=op_linker.fc(fc2, label_cnt, 0.1, activation_func=tf.nn.softmax)
                return final_layer
                #bias changed


def model_generator(input,reuse=False):
    with tf.variable_scope("generator",reuse=reuse):

        #ENCODER
        conv_1 = conv_3d(input, 64, kernerl_size=(11,11, 11), strides=(3, 3, 3), name="conv_1")
        #print("after encoder 1", conv_1.get_shape())
        max_pool_1 = max_pool3D(conv_1, size_pool=(2, 2, 2), stride=(2, 2, 2))
        #print("after max 1", max_pool_1.get_shape())

        conv_2 = conv_3d(max_pool_1, 192, kernerl_size=(2, 2, 2), strides=(2, 2, 2), name="conv_2")
        #print("after encoder 2", conv_2.get_shape())
        max_pool_2 = max_pool3D(conv_2, size_pool=(2, 2, 2), stride=(2, 2, 2))
        #print("after max 2", max_pool_2.get_shape())

        conv_3 = conv_3d(max_pool_2, 384, kernerl_size=(3, 3, 3), strides=(1, 1, 1), name="conv_3",padding="SAME")
        #print("after Encoder 3 ", conv_3.get_shape())
        conv_4 = conv_3d(conv_3, 256, kernerl_size=(2, 2, 2), strides=(1, 1, 1), name="conv_4",padding="SAME")
        #print("after Encoder 4", conv_4.get_shape())

        conv_5 = conv_3d(conv_4, 256, kernerl_size=(2, 3, 3), strides=(2, 3, 3), name="conv_5",padding="SAME")
        #print("after Encoder 5", conv_5.get_shape())
        max_pool_5 = max_pool3D(conv_5, size_pool=(2, 2, 2), stride=(2, 2, 2),pad="SAME")
        #print("after max 5", max_pool_5.get_shape())

        #DECODER
        deconv_1=deconv_3D(max_pool_5,filter=256,kernel=(2,3,3),stride=(2,2,2),pad="SAME")
        #print("after Decoder 5", deconv_1.get_shape())
        deconv_2 = deconv_3D(deconv_1, filter=256, kernel=(2, 2, 2), stride=(2, 2, 2),pad="VALID") # K 4 4
        #print("after Decoder 4", deconv_2.get_shape())


        deconv_3 = deconv_3D(deconv_2, filter=384, kernel=(3, 3, 3), stride=(1, 1, 1))
        #print("after Decoder 3", deconv_3.get_shape())

        deconv_4 = deconv_3D(deconv_3, filter=192, kernel=(2, 2, 2), stride=(2, 2, 2),pad="SAME")
        #print("after Decoder 2", deconv_4.get_shape())
        deconv_5 =K.resize_volumes(deconv_4, 4, 5, 5, "channels_last")# deconv_3D(deconv_4, filter=64, kernel=(11, 11, 11), stride=(4, 4, 4))
        #print("after Decoder 1", deconv_5.get_shape())

        #print("after res 1", net.get_shape())

        reconstructed_image=deconv_3D(deconv_5, filter=1, kernel=(11, 11, 11), stride=(3, 3, 3),pad="VALID")
        #print("shape", reconstructed_image.get_shape())


        return  reconstructed_image


def discriminator(input_feed,reuse=False):
    with tf.variable_scope("discriminator",reuse=reuse):
        conv_1 = conv_3d(input_feed, 64, kernerl_size=(11, 11, 11), strides=(3, 3, 3), name="conv_1")
        max_pool_1 = max_pool3D(conv_1, size_pool=(2, 2, 2), stride=(2, 2, 2))

        conv_2 = conv_3d(max_pool_1, 192, kernerl_size=(2, 2, 2), strides=(2, 2, 2), name="conv_2")
        max_pool_2 = max_pool3D(conv_2, size_pool=(2, 2, 2), stride=(2, 2, 2))

        conv_3 = conv_3d(max_pool_2, 384, kernerl_size=(3, 3, 3), strides=(1, 1, 1), name="conv_3", padding="SAME")
        conv_4 = conv_3d(conv_3, 256, kernerl_size=(2, 2, 2), strides=(1, 1, 1), name="conv_4", padding="SAME")

        conv_5 = conv_3d(conv_4, 256, kernerl_size=(2, 3, 3), strides=(2, 3, 3), name="conv_5", padding="SAME")
        max_pool_5 = max_pool3D(conv_5, size_pool=(2, 2, 2), stride=(2, 2, 2), pad="SAME")


        dense_layer_1=fully_connect(max_pool_5,unit=1000)
        dense_layer_2 = fully_connect(dense_layer_1, unit=1)
        output= tf.nn.sigmoid(dense_layer_2)


        return  output,dense_layer_2




def conv_3d(input,filter,kernerl_size,strides,padding="VALID",activation=tf.nn.relu,name=None,train=True):
    return tf.layers.conv3d(inputs=input, filters=filter, kernel_size=kernerl_size, strides=strides, padding=padding,name=name,trainable=train)



def deconv_3D(input,filter,kernel,stride,activate=tf.nn.relu,pad="SAME"):
    return  tf.layers.conv3d_transpose(inputs=input,filters=filter,kernel_size=kernel,strides=stride,activation=activate,padding=pad)
def max_pool3D (input,size_pool,stride,pad="VALID"):
   return  tf.layers.max_pooling3d(inputs=input,pool_size=size_pool,strides=stride,padding=pad)



def fully_connect(input,unit,activate=tf.nn.leaky_relu):
    return tf.layers.dense(inputs=input,units=unit,activation=activate)


def generator_loss(r_logits,f_logits):

    return  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
        r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))




def  class_training(input_feed,class_num,drop_out=1,reuse=False):

    with tf.variable_scope("generator", reuse=reuse):
        conv_1 = conv_3d(input_feed, 64, kernerl_size=(11, 11, 11), strides=(3, 3, 3), name="conv_1",train=False)
        max_pool_1 = max_pool3D(conv_1, size_pool=(2, 2, 2), stride=(2, 2, 2))

        conv_2 = conv_3d(max_pool_1, 192, kernerl_size=(2, 2, 2), strides=(2, 2, 2), name="conv_2",train=False)
        max_pool_2 = max_pool3D(conv_2, size_pool=(2, 2, 2), stride=(2, 2, 2))

        conv_3 = conv_3d(max_pool_2, 384, kernerl_size=(3, 3, 3), strides=(1, 1, 1), name="conv_3", padding="SAME",train=False)
        conv_4 = conv_3d(conv_3, 256, kernerl_size=(2, 2, 2), strides=(1, 1, 1), name="conv_4", padding="SAME",train=False)

        conv_5 = conv_3d(conv_4, 256, kernerl_size=(2, 3, 3), strides=(2, 3, 3), name="conv_5", padding="SAME",train=False)
        max_pool_5 = max_pool3D(conv_5, size_pool=(2, 2, 2), stride=(2, 2, 2), pad="SAME")

        dense_layer_1 = fully_connect(max_pool_5, unit=2000,)
        drop_out=tf.nn.dropout(dense_layer_1,keep_prob=drop_out)
        dense_layer_2 = fully_connect(drop_out, unit=512)
        dense_layer_3 = fully_connect(dense_layer_2, unit=class_num,activate=tf.nn.softmax)

        return  dense_layer_3


def gan_loss(logits_in,labels_in):
    return   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


def autoencoder(inputs, batch, training):

  with tf.variable_scope('autoencoder') as scope:

     #scope.reuse_variables()
        # encoder
     #print("shape 1",inputs.shape)
     with tf.variable_scope('conv1layer'):
        #print("Main", inputs.shape)
        net = op_linker.conv(inputs, 2,32)  #3
        net=tf.layers.batch_normalization(net, training=training)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        #print("check ", net.shap # e)
        #print("shape 1",net.shape)

     with tf.variable_scope('conv2layer'):
        net = op_linker.conv(net, 2,64)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
        #print("shape 2", net.shape)

     with tf.variable_scope('conv3layer'):
        net = op_linker.conv(net, 2,128)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 3", net.shape)

     with tf.variable_scope('conv4layer'):
        net = op_linker.conv(net, 2,256)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 4", net.shape)

     with tf.variable_scope('conv5layer'):
        net = op_linker.conv(net, 2,512)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME') #tuned
        #print("shape 5", net.shape)

        # decoder
     with tf.variable_scope('decon1layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 256,batch_size=batch)
        net = tf.layers.batch_normalization(net, training=training)
        #print("check ", net.shape)


     with tf.variable_scope('decon2layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 128,batch_size=batch)
        net = tf.layers.batch_normalization(net, training=training)


     with tf.variable_scope('decon3layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 64,batch_size=batch)  # for max pooling , conv_padding='VALID'
        net = tf.layers.batch_normalization(net, training=training)


     with tf.variable_scope('decon4layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 32,batch_size=batch)
        net = tf.layers.batch_normalization(net, training=training)

     with tf.variable_scope('decon5layer'):
        net = K.resize_volumes(net, 2, 2, 2, "channels_last")
        net = op_linker.deconv(net, 2, 1, batch_size=batch)

        return net


def domain_parameters(flip_value):

    with tf.variable_scope('domain_predictor'):
        # Flip the gradient when backpropagating through this operation
        global  shape_to_return
        global  d_W_fc0
        global  d_b_fc0
        l = flip_value
        feature = shape_to_return
        feat = feature   #flip_gradient(feature, l)
        d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

        d_W_fc1 = weight_variable([512, 2])
        d_b_fc1 = bias_variable([2])
        final_la =  tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
        domain_pred = tf.nn.softmax(final_la)
        return domain_pred



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def model_accuracy(logits, labels):
    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def domain_accuracy(logits, labels):
    # accuracy
    with tf.name_scope('domain_accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('domain_accuracy', accuracy)
    return accuracy


def model_loss(logits, labels):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
    return loss



def loss_autoencoder(d_inputs, ae_output):
    with tf.name_scope('autoencoder_loss'):
        loss = tf.reduce_mean(tf.square(ae_output- d_inputs))
        tf.summary.scalar('autoencoder_loss', loss)
    return loss




def domain_loss(logits, labels):
    with tf.name_scope('domain_loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        tf.summary.scalar('domain_loss', loss)
    return loss


def train_rms_prop(loss, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp'):
    return tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon, use_locking, name).minimize(loss)





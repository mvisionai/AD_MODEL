import os
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
import os
from ARCH_3D import Alex3D as model
import numpy as np
from datetime import datetime
from AD_Dataset import Dataset_Import
from tensorflow.contrib.tensorboard.plugins import projector
import AD_Constants as constant
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Main_run(Dataset_Import):

    def __init__(self):
        super().__init__()
        self.now = datetime.now()
        self.training_epoch = 500
        self.gan_epoch = 100
        self.batch_size = 1
        self.validation_interval = 20
        self.keep_prob = 0.50
        self.learn_rate_gan = 0.001
        self.learn_rate_con = 0.001
        self.is_train = True
        self.check_point_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights");
        self.check_point=os.path.join(self.check_point_path,"saved_weights");
        self.train_weight_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_weights");
        self.train_weights=os.path.join(self.train_weight_dir, "weights_trained");
        self.image_depth = constant.img_shape_tuple[0]  # image size
        self.image_height = constant.img_shape_tuple[1]  # image size
        self.image_width = constant.img_shape_tuple[2]  # image size
        self.img_channel = constant.img_channel  # number of channels (1 for black & white)
        self.label_cnt = len(constant.classify_group)  # number of classes

    def gan_training(self):

        real_inputs = tf.placeholder("float",
                                     [None, self.image_depth, self.image_height, self.image_width, self.img_channel],
                                     'inputs')
        learning_rate = tf.placeholder("float", None, name='learning_rate')
        generated_input = tf.placeholder("float", [None, self.image_depth, self.image_height, self.image_width,
                                                   self.img_channel], "gen_input")

        generator = model.generator(generated_input)

        D_output_real, D_logits_real = model.discriminator(real_inputs)
        D_output_fake, D_logits_fake = model.discriminator(generator, reuse=True)

        # LOSSES
        D_real_loss = model.gan_loss(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
        D_fake_loss = model.gan_loss(D_logits_fake, tf.zeros_like(D_logits_real))

        D_loss = D_real_loss + D_fake_loss

        G_loss = model.gan_loss(D_logits_fake, tf.ones_like(D_logits_fake))

        tvars = tf.trainable_variables()

        d_vars = [var for var in tvars if "discriminator" in var.name]
        g_vars = [var for var in tvars if "generator" in var.name]

        # OPTIMIZERS
        D_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=d_vars)
        G_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=g_vars)

        # print([v.name for v in d_vars])

        saver = tf.train.Saver(var_list=g_vars)


        source_1T_data = self.shuffle(self.all_source_data(augment_data=self.data_augmentation))
        source_3T_data = self.all_target_data()
        all_data_gan = self.source_target_validation(source_1T_data, source_3T_data)
        num_batches = int(len(all_data_gan) / self.batch_size)
        self.set_random_seed(np.random.random_integers(1000))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Recall an epoch is an entire run through the training data
            for epoch in range(self.gan_epoch):
                # // indicates classic division
                self.set_random_seed(np.random.random_integers(1000))
                shuffle_data=self.shuffle(all_data_gan)
                for i in range(num_batches):
                    # Grab batch of images
                    batch_images = np.asarray(
                        [data for data in self.next_batch_combined(self.batch_size,shuffle_data)])

                    # Run optimizers, no need to save outputs, we won't use them
                    _, disc_loss = sess.run([D_trainer, D_loss], feed_dict={real_inputs: list(batch_images[0:, 0]),
                                                                            generated_input: list(batch_images[0:, 0]),
                                                                            learning_rate: self.learn_rate_gan})
                    _ = sess.run(G_trainer, feed_dict={generated_input: list(batch_images[0:, 0]),
                                                       learning_rate: self.learn_rate_gan})

                print("Currently on Epoch {} of {} total...".format(epoch + 1, self.gan_epoch))
                print("Epoch Loss {:.6f} ".format(disc_loss))
                print(" ", end="\n")
                saver.save(sess,self.check_point )
                # samples.append(gen_sample)



            #VALIDATION MODEL



    def main_model(self):

        tf.reset_default_graph()
        data_feed = tf.placeholder("float",
                                   [None, self.image_depth, self.image_height, self.image_width, self.img_channel],
                                   'inputs')
        data_labels = tf.placeholder("float", [None, self.label_cnt], 'labels')
        dropout_prob = tf.placeholder("float", None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')

        model_logit = model.class_training(data_feed, self.label_cnt,dropout_prob, reuse=False)

        # ACCURACY
        accuracy = model.accuracy(model_logit, data_labels)

        # LOSSES
        loss = model.loss(model_logit, data_labels)

        # OPTIMIZER
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        saver = tf.train.Saver()

        self.set_random_seed(np.random.random_integers(1000))
        source_1T_data = self.shuffle(self.all_source_data(augment_data=self.data_augmentation))
        source_3T_data = self.all_target_data(augment_data=self.data_augmentation)
        all_available_data = self.source_target_validation(source_1T_data, source_3T_data)

        training_data=self.training_data_source(all_available_data)
        validation_data=self.validation_data_source(all_available_data)

        num_batches = int(len(training_data) / self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Recall an epoch is an entire run through the training data
            latest_ckp = tf.train.latest_checkpoint(self.check_point_path)
            self.saver.restore(sess, latest_ckp)
            for epoch in range(self.gan_epoch):
                # // indicates classic division
                self.set_random_seed(np.random.random_integers(1000))
                shuffle_data=self.shuffle(training_data)
                for i in range(num_batches):
                    # Grab batch of images
                    batch_images = np.asarray(
                        [data for data in self.next_batch_combined(self.batch_size,shuffle_data)])

                    # Run optimizers, no need to save outputs, we won't use them
                    _, train_accuracy, train_loss = sess.run([train_optimizer, accuracy, loss],
                                                              feed_dict={data_feed: list(batch_images[0:, 0]),
                                                                        data_labels: list(batch_images[0:, 1]),
                                                                        dropout_prob:self.keep_prob,
                                                                        learning_rate: self.learn_rate_con})

                print("Currently on Epoch {} of {} total...".format(epoch + 1, self.gan_epoch))
                print("Training Loss {:.6f} ".format(train_loss))
                print("Training Acc. {:.6f} ".format(train_accuracy))
                print(" ", end="\n")



                # Sample from generator as we're training for viewing afterwards

                # gen_sample = sess.run(generator(generated_input, reuse=True), feed_dict={generated_input: sample_z})
                saver.save(sess,self.train_weights)
                # samples.append(gen_sample)


               #MODEL VALIDATION

                _, valid_accuracy, valid_loss = sess.run([accuracy, loss],
                                                       feed_dict={data_feed: list(validation_data[0:, 0]),
                                                                data_labels: list(validation_data[0:, 1]),
                                                           })

                print("Val. Loss {:.6f} ".format(train_loss))
                print("Val. Acc. {:.6f} ".format(train_accuracy))
                print(" ", end="\n")



if __name__ == '__main__':
    run = Main_run()
    run.gan_training()

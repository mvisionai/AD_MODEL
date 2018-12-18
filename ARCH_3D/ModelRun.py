import os
from ARCH_3D.Alex3D import gan_loss, discriminator, class_training, model_accuracy, model_loss, model_generator
import numpy as np
from datetime import datetime
from AD_Dataset import Dataset_Import
import AD_Constants as constant
import time
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = " "


# from ARCH_3D import Alex3D as model

#os.environ["CUDA_VISIBLE_DEVICES"]=""

class Main_run(Dataset_Import):



    def __init__(self, gan_training=True, model_train=False):
        super().__init__()
        self.now = datetime.now()
        self.training_epoch = 500
        self.gan_epoch = 200
        self.train_epoch=200
        self.batch_size = 2
        self.validation_interval = 20
        self.keep_prob = 0.50
        self.learn_rate_gan = 0.001
        self.learn_rate_con = 0.001
        self.train_gan = gan_training
        self.model_train = model_train
        self.is_train = True
        self.check_point_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights");
        self.check_point = os.path.join(self.check_point_path, "saved_weights");
        self.train_weight_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_weights");
        self.train_weights = os.path.join(self.train_weight_dir, "weights_trained");
        self.summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summaries");
        self.gan_summary=os.path.join(self.summary_path, "gan");
        self.train_summary=os.path.join(self.summary_path, "train");
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

        generator = model_generator(generated_input)

        D_output_real, D_logits_real = discriminator(real_inputs)
        D_output_fake, D_logits_fake = discriminator(generator, reuse=True)

        # LOSSES
        D_real_loss = gan_loss(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
        D_fake_loss = gan_loss(D_logits_fake, tf.zeros_like(D_logits_real))

        D_loss = D_real_loss + D_fake_loss
        tf.summary.scalar('discriminator_loss',D_loss)

        G_loss = gan_loss(D_logits_fake, tf.ones_like(D_logits_fake))

        tvars = tf.trainable_variables()

        d_vars = [var for var in tvars if "discriminator" in var.name]
        g_vars = [var for var in tvars if "generator" in var.name]

        # OPTIMIZERS

        D_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=d_vars)
        G_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=g_vars)


        #self.train_gan_opt=tf.group(G_trainer,D_trainer)

        # print([v.name for v in d_vars])

        saver = tf.train.Saver(var_list=g_vars)
        merged_summary = tf.summary.merge_all()

        source_1T_data = self.shuffle(self.all_source_data(augment_data=self.data_augmentation))
        source_3T_data = self.all_target_data(augment_data=self.data_augmentation)
        all_data_gan = self.source_target_validation(source_1T_data, source_3T_data)
        total_data_length=len(all_data_gan)
        num_batches = int(total_data_length/ self.batch_size)

        print("batch_num ", num_batches)
        print("total num ", len(all_data_gan))

        # for j in range(num_batches):
        #    datas = self.next_batch_combined_gan(self.batch_size, all_data_gan)
        #    sd= [data for data in datas]
        #    print(np.array(sd))
        #    #self.show_image()



        with tf.Session() as sess:
            gan_sum_writer = tf.summary.FileWriter(self.gan_summary,sess.graph)
            sess.run(tf.global_variables_initializer())
            # Recall an epoch is an entire run through the training data   self.gan_epoch
            for epoch in range(self.gan_epoch):
                # // indicates classic division
                self.set_random_seed(np.random.random_integers(1000))
                shuffle_data =  self.shuffle(all_data_gan)
                counter=0   #num_batches

                print("Currently on Epoch {} of {} total...".format(epoch + 1, self.gan_epoch))
                for i in range(num_batches):
                    # Grab batch of images

                    feed = self.next_batch_combined_gan(self.batch_size,shuffle_data[counter:counter + self.batch_size])
                    batch_images = [data for data in feed]

                    # Run optimizers, no need to save outputs, we won't use them
                    _ = sess.run(G_trainer, feed_dict={generated_input: batch_images,
                                                       learning_rate: self.learn_rate_gan})

                    _, disc_loss,gan_summary = sess.run([D_trainer, D_loss,merged_summary],
                                            feed_dict={real_inputs:batch_images,
                                                       generated_input:batch_images,
                                                       learning_rate: self.learn_rate_gan})

                    _ = sess.run(G_trainer, feed_dict={generated_input:batch_images,
                                                       learning_rate: self.learn_rate_gan})
                    counter = (counter + self.batch_size) % total_data_length
                    print('GAN Epoch %d/%d, batch %d/%d is finished!' % (epoch+1,self.gan_epoch,i+1,num_batches))
                    print("\t\tGAN Loss: {:.6f}".format(disc_loss))


                print("Epoch Loss {:.6f} ".format(disc_loss))
                print(" ", end="\n")
                gan_sum_writer.add_summary(gan_summary, global_step=epoch)
                saver.save(sess, self.check_point)
                # samples.append(gen_sample)

            # VALIDATION MODEL

    def main_model(self):

        tf.reset_default_graph()
        data_feed = tf.placeholder("float",
                                   [None, self.image_depth, self.image_height, self.image_width, self.img_channel],
                                   'inputs')
        data_labels = tf.placeholder("float", [None, self.label_cnt], 'labels')
        dropout_prob = tf.placeholder("float", None, name='keep_prob')
        learning_rate = tf.placeholder("float", None, name='learning_rate')

        model_logit = class_training(data_feed, self.label_cnt, dropout_prob, reuse=False)

        # ACCURACY
        accuracy = model_accuracy(model_logit, data_labels)

        # LOSSES
        loss = model_loss(model_logit, data_labels)

        # OPTIMIZER
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


        train_merged_summary = tf.summary.merge_all()

        self.set_random_seed(np.random.random_integers(1000))
        source_1T_data = self.shuffle(self.all_source_data(augment_data=self.data_augmentation))
        source_3T_data = self.all_target_data(augment_data=self.data_augmentation)
        all_available_data = self.source_target_validation(source_1T_data, source_3T_data)

        training_data = self.training_data_source(all_available_data)
        validation_data = self.validation_data_source(all_available_data)
        total_data_length=len(training_data)
        num_batches = int(total_data_length / self.batch_size)

        tvars = tf.trainable_variables()
        res_vars = [var for var in tvars if "generator" in var.name]

        print(res_vars)



        with tf.Session() as sess:
            train_sum_writer = tf.summary.FileWriter(self.train_summary, sess.graph)
            sess.run(tf.global_variables_initializer())
            tf.stop_gradient('generator/conv_5/kernel:0')
            # Recall an epoch is an entire run through the training data
            #latest_ckp = tf.train.latest_checkpoint(self.check_point_path)
            #print(latest_ckp)
            #saver.restore(sess, latest_ckp)
            res_saver= tf.train.import_meta_graph(os.path.join(self.check_point_path,"saved_weights.meta"))
            res_saver.restore(sess,self.check_point)

            #print(sess.run('generator/conv_5/kernel:0'))


            #exit(1)
            for epoch in range(2):
                # // indicates classic division
                self.set_random_seed(np.random.random_integers(1000))
                shuffle_data = self.shuffle(training_data)
                counter = 0

                print("Currently on Epoch {} of {} total...".format(epoch + 1, self.train_epoch))
                for i in range(2):
                    # Grab batch of images
                    batch_images = np.asarray([data for data in self.next_batch_combined(self.batch_size,shuffle_data[counter:counter + self.batch_size])])

                    # Run optimizers, no need to save outputs, we won't use them
                    _, train_accuracy, train_loss,train_summary = sess.run([train_optimizer, accuracy, loss,train_merged_summary],
                                                             feed_dict={data_feed: list(batch_images[0:, 0]),
                                                                        data_labels: list(batch_images[0:, 1]),
                                                                        dropout_prob: self.keep_prob,
                                                                        learning_rate: self.learn_rate_con})

                    counter = (counter + self.batch_size) % total_data_length
                    print('Training Epoch %d/%d, batch %d/%d is finished!' % (epoch + 1, self.gan_epoch, i + 1, num_batches))
                    print("\t\tTraining Acc: {:.6f}".format(train_accuracy))
                    print("\t\tTraining Loss: {:.6f}".format(train_accuracy))

                print("Training Loss {:.6f} ".format(train_loss))
                print("Training Acc. {:.6f} ".format(train_accuracy))
                print(" ", end="\n")
                #saver.save(sess, self.train_weights)
                train_sum_writer.add_summary(train_summary, global_step=epoch)
                # samples.append(gen_sample)



                # MODEL VALIDATION

                _, valid_accuracy, valid_loss = sess.run([accuracy, loss],
                                                         feed_dict={data_feed: list(validation_data[0:, 0]),
                                                                    data_labels: list(validation_data[0:, 1]),
                                                                    })

                print("Val. Loss {:.6f} ".format(train_loss))
                print("Val. Acc. {:.6f} ".format(train_accuracy))
                print(" ", end="\n")


if __name__ == '__main__':

    try:
        run = Main_run(gan_training=True, model_train=False)

        if run.train_gan:
            run.gan_training()

        if run.model_train:
           run.main_model()


    except Exception as e:
        print(e)
        raise

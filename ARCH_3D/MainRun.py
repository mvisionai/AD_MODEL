import os
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
import os
import  Alex3D as model
import numpy as np
from datetime import datetime
from AD_Dataset import  Dataset_Import
from tensorflow.contrib.tensorboard.plugins import projector
import AD_Constants as constant
from ARCH_3D  import ops as op_linker
import time

#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
class Main_run(Dataset_Import):

    def __init__(self):
        super().__init__()
        self.now = datetime.now()
        self.training_epoch =20
        self.auto_encode_epoch = 20
        self.batch_size = 2
        self.validation_interval = 20
        self.dropout_prob = 0.50
        self.learn_rate_auto = 0.01
        self.learn_rate_pred = 0.01
        self.learning_rate_convlayer=0.01
        self.learn_rate_fully_layer=0.01
        self.rms_decay = 0.90
        self.weight_decay = 0.06
        self.is_train = True
       # 0.0005, 0.001, 0.00146


        # Data info.
        self.image_size =constant.img_shape_tuple[0]  # image size
        self.img_channel = constant.img_channel  # number of channels (1 for black & white)
        self.label_cnt = len(constant.classify_group)  # number of classes



    def auto_encoder_platform(self,train_type):

        if train_type == "domain":
            train_list = self.source_target_combined()

        elif train_type == "single":
            train_list = self.source_data_feed()

        return train_list


    def training_platform(self,train_type,source,target):
        if train_type == "domain":
            train_list = self.source_target_combined()

        elif train_type == "single":
            train_list = self.source_data_feed()

        return train_list

    def train(self, train_type: str="domain", use_encoder_saver: bool=False, use_train_saver: bool=False,batch_validation:bool=False):

            # build graph

            with tf.device("/cpu:0"):

                    config=tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

                    if train_type  not in ["domain","single"]:
                        raise("Train type argument invalid")




                    if use_encoder_saver == False:




                        if train_type == "domain":

                            inputs, labels, training, dropout_keep_prob, learning_rate, domain_label, flip_grad = model.input_placeholder(
                                self.image_size, self.img_channel, self.label_cnt, train_type)

                        elif train_type == "single":
                            inputs, labels, training, dropout_keep_prob, learning_rate, flip_grad = model.input_placeholder(
                                self.image_size, self.img_channel, self.label_cnt, train_type)




                        autoencoder_run=model.autoencoder(inputs,self.batch_size, training)

                        # autoencoder loss
                        autoencoder_loss = model.loss_autoencoder(inputs, autoencoder_run)


                        #auto_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        #with tf.control_dependencies(auto_update_ops):

                        train_autoencode = tf.train.MomentumOptimizer(learning_rate,momentum=self.rms_decay).minimize(autoencoder_loss)



                        saver = tf.train.Saver()



                        merged_summary = tf.summary.merge_all()

                        with tf.Session(config=config) as encoder_sess:
                            sum_writer=tf.summary.FileWriter(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "summaries"+os.sep+"auto"),encoder_sess.graph)
                            encoder_sess.run(tf.global_variables_initializer())
                            #self.training_epoch

                            print("Autoencoder Pretraining ...",end="\n\n")



                            for encoder_epoch in range(40):

                              start_time = time.time()

                              train_list = self.auto_encoder_platform(train_type)
                              total_batch_auto = int(len(train_list) / self.batch_size)
                              for i in range(total_batch_auto):

                                    feed=self.next_batch_combined_encoder(self.batch_size,train_list)
                                    input_feed= [i for i in feed]

                                    auto_encode_feed_dict = {inputs:input_feed,learning_rate: self.learn_rate_auto,
                                                             training:True, dropout_keep_prob:self.dropout_prob}
                                    _,encoder_loss,summary_out=encoder_sess.run([train_autoencode,autoencoder_loss,merged_summary],feed_dict=auto_encode_feed_dict)

                                    print('Auto Epoch %d/%d, batch %d/%d is finished!'%(encoder_epoch, self.auto_encode_epoch, i, total_batch_auto))
                                    print("Reconstruction Loss: {:.6f}".format(encoder_loss))
                                    #saver.save(encoder_sess, 'np_data/model_iter', global_step=i)
                              sum_writer.add_summary(summary_out,global_step=encoder_epoch)
                              end_time = time.time()
                              print(" ", end="\n")
                              print("Epoch " + str(encoder_epoch + 1) + " completed : Time usage " + str(int(end_time - start_time)) + " seconds")
                              print("Reconstruction Loss: {:.6f}".format(encoder_loss))
                              print(" ", end="\n")
                              self.i=0

                              # Save the final model
                            saver.save(encoder_sess, 'np_data/model_final')



                    tf.reset_default_graph()

                    g2 = tf.Graph()
                    # Merge all summaries together
                    source_data = self.all_source_data()
                    target_data = self.all_target_data()

                    source_training_data = self.training_data_source(source_data)
                    source_validation_data = self.validation_data_source(source_data)

                    target_training_data = self.training_data_target(target_data)
                    target_validation_data = self.validation_data_target(target_data)


                    with g2.as_default() as g3:


                            inputs, labels, training, dropout_keep_prob, learning_rate, domain_label, flip_grad =  model.input_placeholder(self.image_size, self.img_channel, self.label_cnt)


                            logits = model.vgg16(inputs, training, dropout_keep_prob, self.label_cnt)

                            accuracy = model.accuracy(logits, labels)
                            pred_loss = model.loss(logits, labels)

                            if train_type=="domain":

                                domain_logits = model.domain_parameters(flip_grad)
                                domain_accuracy = model.domain_accuracy(domain_logits, domain_label)
                                domain_loss = model.domain_loss(domain_logits, domain_label)
                                total_loss = pred_loss + domain_loss * 0.01
                                fully_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/" + fc_var)
                                                   for fc_var in
                                                   constant.not_trained]
                            elif train_type=="single":
                                total_loss = pred_loss * 0.01
                                fully_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/" + fc_var)
                                                   for fc_var in
                                                   constant.not_trained if fc_var != "domain_predictor"]

                            convolve_variables = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution/"+con_var) for con_var in
                                                  constant.pre_trained]




                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
                            with tf.control_dependencies(update_ops):
                                # train = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
                                train_op_trained = tf.train.GradientDescentOptimizer(self.learning_rate_convlayer).minimize(total_loss)#var_list=convolve_variables

                                train_op_untrained = tf.train.GradientDescentOptimizer(self.learn_rate_fully_layer).minimize(total_loss)# var_list=fully_variables

                                train_op = tf.group(train_op_trained, train_op_untrained)

                            tr_merge_summary = tf.summary.merge_all()


                   # tf.reset_default_graph()





                    with tf.Session(graph=g3,config=config) as  tr_sess:

                        if use_train_saver == False:
                            #imported_meta = tf.train.import_meta_graph("np_data/model_final.meta")
                            tr_writer = tf.summary.FileWriter(
                            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "summaries"+os.sep+"train"),tr_sess.graph)

                            tr_sess.run(tf.global_variables_initializer())
                            train_saver = tf.train.Saver()

                            # imported_meta.restore(tr_sess, tf.train.latest_checkpoint('np_data'))
                            # latest_ckp = tf.train.latest_checkpoint('np_data')
                            #
                            # reader =NewCheckpointReader(latest_ckp)
                            # var_to_shape_map = reader.get_variable_to_shape_map()
                            #
                            # for f in var_to_shape_map:
                            #     print(f,end="\n")
                            #     weight_r = tr_sess.run( ":".join([f, "0"]))
                            #     print(weight_r,end="\n")

                            #load autoencoder pretrained weights and biase
                            #op_linker.load_initial_weights(tr_sess, var_to_shape_map, use_pretrain=True)

                            #exit(0)


                            print(" ", end="\n")
                            print("Initializing Class Training")
                            print(" ", end="\n")


                            for epoch in range(self.training_epoch):

                              start_time_2=time.time()


                              data_list = self.source_target_data(source_training_data,target_training_data,train_type)
                              total_batch = int(len(data_list) / self.batch_size)

                              for i in range(total_batch):

                                    source_feed=self.next_batch_combined(self.batch_size,data_list)

                                    data_source = [data for data in source_feed]
                                    data_source = np.array(data_source)
                                    data_feed=list(data_source[0:, 0])
                                    data_label=list(data_source[0:, 1])




                                    if train_type=="domain":
                                       label_domain = list(data_source[0:, 2])




                                       feed_dict_source_batch = {inputs: data_feed, labels: data_label, training:True,
                                                              dropout_keep_prob: self.dropout_prob, learning_rate: self.learn_rate_pred,
                                                              domain_label: label_domain, flip_grad: 1.0}
                                       _, loss_source_batch, acc_source_batch, accuracy_domain, loss_domain, log_d,acc_log,summary_out2= tr_sess.run(
                                        [train_op, total_loss, accuracy, domain_accuracy, domain_loss, domain_logits,logits,tr_merge_summary],
                                        feed_dict=feed_dict_source_batch)

                                    elif train_type=="single":
                                        feed_dict_source_batch = {inputs: data_feed, labels: data_label, training: True,
                                                                  dropout_keep_prob: self.dropout_prob,
                                                                  learning_rate: self.learn_rate_pred,
                                                                  flip_grad: 1.0}

                                        _, loss_source_batch, acc_source_batch, acc_log, summary_out2 = tr_sess.run(
                                            [train_op, total_loss, accuracy,logits,
                                             tr_merge_summary],
                                            feed_dict=feed_dict_source_batch)


                                    #print(data_label)
                                    #print(acc_log)
                                    print('Epoch %d/%d, batch %d/%d is finished!' % (epoch, self.training_epoch, i, total_batch))

                              tr_writer.add_summary(summary_out2,epoch)



                              end_time_2 = time.time()


                              print(" ",end="\n")
                              print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time_2 - start_time_2)) + " seconds")
                              print("Training total_loss: {:.5f}".format(loss_source_batch))
                              print("Training class_accuracy: {0:.3f}".format(acc_source_batch))
                              print(" ", end="\n")


                              if train_type == "domain":
                                print("Training domain_loss: {:.5f}".format(loss_domain))
                                print("Training domain_accuracy: {0:.3f}".format(accuracy_domain))


                              train_saver.save(tr_sess,"train_session/train_model")
                              self.i=0


                          # validation batch

                        if use_train_saver==True:
                            print("Validating Data from Saved Model ",end="\n")
                            train_saver = tf.train.Saver()
                            train_saver.restore(tr_sess,  tf.train.latest_checkpoint('train_session'))




                        if train_type == "domain":

                          len_svalidation = len(source_validation_data)
                          len_tvalidation = len(target_validation_data)
                          max_iteration = int((len_svalidation+len_tvalidation)/self.batch_size)
                          data_list_valid=self.source_target_validation(source_validation_data,target_validation_data)

                        elif train_type == "single":
                            max_iteration= int(len(source_validation_data) /self.batch_size)


                        v_accuracy_list=[]
                        v_accuracyDomain_list = []



                        for steps in range(max_iteration):



                              if train_type == "domain":


                                  vtarget_feed = self.convert_validation_target_data(self.batch_size,data_list_valid)
                                  data_vtarget = [datav for datav in vtarget_feed]
                                  data_vtarget = np.array(data_vtarget)
                                  validation_target_dataset = list(data_vtarget[0:, 0])
                                  valid_target_label = list(data_vtarget[0:, 1])
                                  valid_target_d_label = list(data_vtarget[0:, 2])



                                  # vsource_feed = self.convert_validation_source_data(self.batch_size, source_validation_data)
                                  #
                                  # data_vsource = [data for data in vsource_feed]
                                  # data_vsource = np.array(data_vsource)
                                  # validation_source_dataset = list(data_vsource[0:, 0])
                                  # valid_source_label = list(data_vsource[0:, 1])
                                  # valid_source_d_label = list(data_vsource[0:, 2])
                                  #
                                  #
                                  #
                                  # acc_target_valid = tr_sess.run(accuracy,
                                  #                                feed_dict={inputs: validation_target_dataset,
                                  #                                           labels: valid_target_label,
                                  #                                           training: False,
                                  #                                           dropout_keep_prob: 1.0})
                                  #
                                  # valid_data_feed = np.vstack([validation_source_dataset, validation_target_dataset])
                                  # valid_data_label = np.vstack([valid_source_label, valid_target_label])
                                  # valid_data_d_label = np.vstack([valid_source_d_label, valid_target_d_label])

                                  validation_accuracy, acc_domain = tr_sess.run([accuracy, domain_accuracy],
                                                                                feed_dict={inputs: validation_target_dataset,
                                                                                           labels: valid_target_label,
                                                                                           training: False,
                                                                                           dropout_keep_prob: 1.0,
                                                                                           domain_label: valid_target_d_label,
                                                                                           flip_grad: 0})






                              elif train_type == "single":

                                  vsource_feed = self.convert_validation_source_data(self.batch_size, source_validation_data)

                                  data_vsource = [data for data in vsource_feed]
                                  data_vsource = np.array(data_vsource)
                                  validation_source_dataset = list(data_vsource[0:, 0])
                                  valid_source_label = list(data_vsource[0:, 1])

                                  validation_accuracy = tr_sess.run(accuracy,
                                                             feed_dict={inputs: validation_source_dataset,
                                                                        labels: valid_source_label,
                                                                        training: False,
                                                                        dropout_keep_prob: 1.0})












                              print(" ", end="\n")
                              print("-------Validation ",steps+1,"----------", end="\n")
                              print("Validation   accuracy: {0:.2f}".format(validation_accuracy))

                              v_accuracy_list.append(round(validation_accuracy,2))


                              if train_type == "domain":
                                #print("Validation source  accuracy: {0:.2f}".format(acc_source_valid))
                                #print("Validation target  accuracy: {0:.2f}".format(acc_target_valid))
                                print("Validation domain  accuracy: {0:.2f}".format(acc_domain))
                                v_accuracyDomain_list.append(round(acc_domain, 2))




                        print(" ",end="\n")
                        print("Average Validation Accuracy ",round(sum(v_accuracy_list)/len(v_accuracy_list),2))

                        if train_type == "domain":
                          print("Average Domain Accuracy ", round(sum(v_accuracyDomain_list) / len(v_accuracyDomain_list), 2))
                              # end of validation batch


def netTrain(net_type:str=None):
    pass

if __name__ == '__main__':

  #try:
   run_train = Main_run()
   run_train.train(train_type="single", use_encoder_saver=True, use_train_saver=False)

  #except Exception as ex:
    #print("Exeception caught ",ex)
    #raise






import os
import cv2
import numpy as np
from model import network
from data_gen import read_and_decode

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


tf.app.flags.DEFINE_string('mode', None,'whether train or test')




class Sliding_Convolution(object):
    def __init__(self,is_train=False):
        self.class_num=5990
        self.character_step = 8
        self.character_height = 32
        self.character_width = 32

        #train
        self.train_tfrecords_name="./make_tfrecords/data.tfrecords"
        self.summary_save_path="./summary/"
        self.summary_steps=10000
        self.save_steps=10
        self.save_path="./save/"

        #test:
        self.model_path="./save/sliding_conv.ckpt-10"

        if is_train:
            self.batch_size = 16
            self.with_clip = True
            self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                   character_height=self.character_height, character_width=self.character_width,
                                   character_step=self.character_step, is_train=True)
        else:
            current_path = os.path.dirname(os.path.abspath(__file__))
            self.char_dict = self.create_char_dict(
                os.path.join(current_path, "char_std_5990.txt"))


            self.batch_size = 1
            self.graph = tf.Graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
            with self.session.as_default():
                with self.graph.as_default():
                    self.network = network(batch_size=self.batch_size, class_num=self.class_num,
                                        character_height=self.character_height, character_width=self.character_width,
                                        character_step=self.character_step, is_train=False)

                    self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"], self.network["seq_len"],
                                                                 merge_repeated=True)

                    init = tf.global_variables_initializer()
                

                    self.session.run(init)
                    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
                    saver.restore(self.session, self.model_path)


    def create_char_dict(self,filename):
        char_dict = {}
        file = open(filename, 'r', encoding='utf-8').read().splitlines()
        index = 0
        for char in file:
            char_dict[index] = char[0]
            index += 1
        return char_dict

    def adjust_label(self, result):
        result_str = ""

        for x, char in enumerate(result):
            result_str += self.char_dict[char]
        return result_str

    def train_net(self):
        train_data, train_label = read_and_decode(self.train_tfrecords_name)
        train_inputs, train_targets = tf.train.shuffle_batch([train_data, train_label],
                                                             batch_size=self.batch_size, capacity=2000,
                                                             min_after_dequeue=1000)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3,
                                                   global_step,
                                                   10000,
                                                   0.9,
                                                   staircase=True)


        targets = tf.sparse_placeholder(tf.int32, name='targets')
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=self.network["outputs"], sequence_length=self.network["seq_len"]))
        tf.summary.scalar("loss", loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies([tf.group(*update_ops)]):
                if self.with_clip == True:
                    print("clip grads")
                    tvars = tf.trainable_variables()
                    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
                    optimizer_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
                else:
                    optimizer_op = optimizer.minimize(loss, global_step=global_step)
        else:
            if self.with_clip == True:
                tvars = tf.trainable_variables()
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
                optimizer_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
            else:
                optimizer_op = optimizer.minimize(loss, global_step=global_step)

        
        decoded, log_prob = tf.nn.ctc_greedy_decoder(self.network["outputs"],self.network["seq_len"], merge_repeated=True)
        acc = 1 - tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
        tf.summary.scalar("accuracy", acc)
        merge_summary = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(init)
            threads = tf.train.start_queue_runners(sess=session)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            #saver.restore(session, "./save/sliding_conv.ckpt-160000")
            summary_writer = tf.summary.FileWriter(self.summary_save_path, session.graph)

            while True:
                step_input, step_target, steps, lr = session.run(
                    [train_inputs, train_targets, global_step, learning_rate])

                feed = {self.network["inputs"]: step_input, targets: step_target}
                batch_acc, batch_loss, _ = session.run([acc, loss, optimizer_op], feed)
                print("step is: {}, batch loss is: {} learningrate is {} acc is {}".format(steps, batch_loss, lr,batch_acc))
                
                if steps > 0 and steps % self.summary_steps == 0:
                    _,batch_summarys = session.run([optimizer_op,merge_summary], feed)
                    summary_writer.add_summary(batch_summarys, steps)
                if steps > 0 and steps % self.save_steps == 0:
                    save_path = saver.save(session, os.path.join(self.save_path,"sliding_conv.ckpt"), global_step=steps)
                    print(save_path)

    def test_net(self,input_data):
        if input_data.ndim==3:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        height, width = input_data.shape
        ratio = height / 32.0
        input_data = cv2.resize(input_data, (32, int(width / ratio)))
        input_data = input_data.reshape((1, 32, int(width / ratio), 1))
        scaled_data = np.asarray(input_data / np.float32(255) -
                                 np.float32(0.5))


        with self.session.as_default():
            with self.graph.as_default():
                feed = {self.network["inputs"]: scaled_data}
                outputs,decoded=self.session.run([self.network["outputs"],self.decoded[0]],feed)
                print(np.max(outputs[0][0]),np.min(outputs[0][0]))
                print(decoded)
                result_str=self.adjust_label(decoded[0])
                return result_str

def main(_):
    if tf.app.flags.FLAGS.mode=="train":
        #train
        sc=Sliding_Convolution(is_train=True)
        sc.train_net()
    else:
        #test
        image=cv2.imread("./make_tfrecords/images/30094265_4150691021.jpg",1)
        sc=Sliding_Convolution(is_train=False)
        result_str=sc.test_net(image)
        print("result:",result_str)
if __name__ == '__main__':
    tf.app.run()

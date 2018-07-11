import cv2
import numpy as np
import tensorflow as tf

def sliding_generate_batch_layer(inputs,character_width=32,character_step=8):
    # inputs: batches*32*280*1

    for b in range(inputs.shape[0]):
        batch_input=inputs[b,:,:,:].reshape((1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))
        for w in range(0,batch_input.shape[2]-character_width,character_step):
            if w==0:
                output_batch=batch_input[:,:,w:(w+1)*character_width,:]
            else:
                output_batch=np.concatenate((output_batch,batch_input[:,:,w:w+character_width,:]),axis=0)

        if b==0:
            output=output_batch
        else:
            output=np.concatenate((output,output_batch),axis=0)
    return output

def network(batch_size=1,class_num=5990,character_height=32,character_width=32,character_step=8, is_train=False):
    network={}

    network["inputs"] =tf.placeholder(tf.float32, [batch_size, 32, None, 1], name='inputs')
    network["seq_len"] = tf.multiply(tf.ones(shape=[batch_size], dtype=tf.int32),tf.floordiv(tf.shape(network["inputs"])[2] - character_width,character_step))

    network["inputs_batch"] = tf.py_func(sliding_generate_batch_layer,[network["inputs"] ,character_width,character_step],tf.float32)
    network["inputs_batch"]=tf.reshape(network["inputs_batch"],[-1,character_height,character_width,1])

    network["conv1"] = tf.layers.conv2d(inputs=network["inputs_batch"], filters=50, kernel_size=(3, 3), padding="same", activation=None)
    network["batch_norm1"] = tf.contrib.layers.batch_norm(
        network["conv1"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm1"] = tf.nn.relu(network["batch_norm1"])
    network["conv2"] = tf.layers.conv2d(inputs=network["batch_norm1"], filters=100, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
    network["dropout2"]=tf.layers.dropout(inputs=network["conv2"] ,rate=0.1)
    network["conv3"] = tf.layers.conv2d(inputs=network["dropout2"] , filters=100, kernel_size=(3, 3), padding="same", activation=None)
    network["dropout3"]= tf.layers.dropout(inputs=network["conv3"] , rate=0.1)
    network["batch_norm3"] = tf.contrib.layers.batch_norm(
        network["dropout3"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm3"]= tf.nn.relu(network["batch_norm3"])
    network["pool3"] = tf.layers.max_pooling2d(inputs=network["batch_norm3"], pool_size=[2, 2], strides=2)
    network["conv4"] = tf.layers.conv2d(inputs=network["pool3"], filters=150, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout4"] = tf.layers.dropout(inputs=network["conv4"], rate=0.2)
    network["batch_norm4"] = tf.contrib.layers.batch_norm(
        network["dropout4"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm4"] = tf.nn.relu(network["batch_norm4"])
    network["conv5"] = tf.layers.conv2d(inputs=network["batch_norm4"], filters=200, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout5"] = tf.layers.dropout(inputs=network["conv5"], rate=0.2)
    network["conv6"] = tf.layers.conv2d(inputs=network["dropout5"], filters=200, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout6"] = tf.layers.dropout(inputs=network["conv6"], rate=0.2)
    network["batch_norm6"] = tf.contrib.layers.batch_norm(
        network["dropout6"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm6"] = tf.nn.relu(network["batch_norm6"])
    network["pool6"] = tf.layers.max_pooling2d(inputs=network["batch_norm6"], pool_size=[2, 2], strides=2)
    network["conv7"] = tf.layers.conv2d(inputs=network["pool6"], filters=250, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout7"] = tf.layers.dropout(inputs=network["conv7"], rate=0.3)
    network["batch_norm7"] = tf.contrib.layers.batch_norm(
        network["dropout7"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm7"] = tf.nn.relu(network["batch_norm7"])
    network["conv8"] = tf.layers.conv2d(inputs=network["batch_norm7"], filters=300, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout8"] = tf.layers.dropout(inputs=network["conv8"], rate=0.3)
    network["conv9"] = tf.layers.conv2d(inputs=network["dropout8"], filters=300, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout9"] = tf.layers.dropout(inputs=network["conv9"], rate=0.3)
    network["batch_norm9"] = tf.contrib.layers.batch_norm(
        network["dropout9"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm9"] = tf.nn.relu(network["batch_norm9"])
    network["pool9"] = tf.layers.max_pooling2d(inputs=network["batch_norm9"], pool_size=[2, 2], strides=2)
    network["conv10"] = tf.layers.conv2d(inputs=network["pool9"], filters=350, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout10"] = tf.layers.dropout(inputs=network["conv10"], rate=0.4)
    network["batch_norm10"] = tf.contrib.layers.batch_norm(
        network["dropout10"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm10"] = tf.nn.relu(network["batch_norm10"])
    network["conv11"] = tf.layers.conv2d(inputs=network["batch_norm10"], filters=400, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.relu)
    network["dropout11"] = tf.layers.dropout(inputs=network["conv11"], rate=0.4)
    network["conv12"] = tf.layers.conv2d(inputs=network["dropout11"], filters=400, kernel_size=(3, 3), padding="same",
                                        activation=None)
    network["dropout12"] = tf.layers.dropout(inputs=network["conv12"], rate=0.4)
    network["batch_norm12"] = tf.contrib.layers.batch_norm(
        network["dropout12"],
        decay=0.9,
        center=True,
        scale=True,
        epsilon=0.001,
        is_training=is_train)
    network["batch_norm12"] = tf.nn.relu(network["batch_norm12"])
    #2*2*400
    network["pool12"] = tf.layers.max_pooling2d(inputs=network["batch_norm12"], pool_size=[2, 2], strides=2)
    network["flatten"] =tf.contrib.layers.flatten(network["pool12"])

    network["fc1"] =tf.contrib.layers.fully_connected(inputs=network["flatten"],num_outputs=900,activation_fn=tf.nn.relu)
    network["dropout_fc1"] = tf.layers.dropout(inputs=network["fc1"], rate=0.5)
    network["fc2"] = tf.contrib.layers.fully_connected(inputs=network["dropout_fc1"], num_outputs=200,activation_fn=tf.nn.relu)
    if is_train:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num,activation_fn=None)
    else:
        network["fc3"] = tf.contrib.layers.fully_connected(inputs=network["fc2"], num_outputs=class_num,activation_fn=tf.nn.sigmoid)
    network["outputs"]=tf.reshape(network["fc3"],[batch_size,-1,class_num])
    network["outputs"]=tf.transpose(network["outputs"],(1,0,2))


    return network


if __name__=="__main__":
    image=cv2.imread("./make_tfrecords/images/30094265_4150691021.jpg",0)
    image=np.reshape(image,[1,image.shape[0],image.shape[1],1])
    image=np.concatenate((image,image),axis=0)
    image = np.concatenate((image, image), axis=0)
    sliding_generate_batch_layer(inputs=image, batches=1)

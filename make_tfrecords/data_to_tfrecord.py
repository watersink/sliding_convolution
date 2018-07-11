import os
import tensorflow as tf
import numpy as np
from PIL import Image


def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        os.remove(filename)
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img_name in enumerate(images):
        print(index)
        img = Image.open(img_name)
        img = img.convert('L')

        img = img.resize((280, 32))
        img_raw = img.tobytes()
        label = np.asarray(labels[index], dtype=np.int64)
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()



def read_data_from_paths(file_path_list, name_list):
    labels = []
    file_names = []

    for i,file_path in enumerate(file_path_list):
        file_name = os.path.join(file_path, name_list[i])
        train_txt = open(file_name)

        for idx in train_txt:
            idx=idx.rstrip("\n")
            spt = idx.split(' ')
            file_names.append(os.path.join(file_path, spt[0]))
            labels.append(spt[1:])


    return file_names, labels

import random
if __name__=="__main__":
    file_path = ["./images/"]
    name = ["label.txt"]
    file_names, labels=read_data_from_paths(file_path, name)

    #-----shuffle------
    filename_label=[]
    for i in range(len(file_names)):
        filename_label.append(i)
        print(i)		
    random.shuffle(filename_label)
    print(filename_label)	
    file_names_new=[]
    labels_new=[]
    for i in filename_label:
        file_names_new.append(file_names[i])
        labels_new.append(labels[i])
    # -----shuffle------
    print("shuffle OK！",len(file_names_new),len(labels_new))


    tfrecord_name="data.tfrecords"
    data_to_tfrecord(file_names_new, labels_new, tfrecord_name)

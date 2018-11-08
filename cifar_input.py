import numpy as np
import pickle
import tensorflow as tf

data_route = 'cifar-10-batches-py'


def unpickle(file):
    with open(file, 'rb') as fp:
        return pickle.load(fp, encoding='latin1')


def read_img(is_test):
    label_dict = {}
    if is_test == 0:
        for i in range(1, 6):
            data_name = data_route + "/data_batch_" + str(i)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径
            data_dict = unpickle(data_name)
            # print(data_dict['label'][3])
            #print(type(data_dict['data'][5000]))
            for j in range(0, 10000):
                img = np.reshape(data_dict['data'][j], (3, 32, 32))  # ['data']为图片二进制数据
                # print(img)
                if data_dict['labels'][j] in list(label_dict.keys()):
                    # print(data_dict['labels'][j])
                    # print(label_dict)
                    # list(label_dict.keys())
                    p = label_dict[data_dict['labels'][j]]
                    # print(p)
                    p.append(img.tolist())
                    label_dict.update({data_dict['labels'][j]: p})
                else:
                    p = [ ]
                    p.append(img.tolist())
                    label_dict.update({data_dict['labels'][j]: p})
        return label_dict
    else:
        test_dict = unpickle("test_batch")
        for i in range(0, 10000):
            img = np.reshape(test_dict['data'][i], (3, 32, 32))  # ['data']为图片二进制数据
            if test_dict['labels'][i] in list(label_dict.keys()):
                label_dict[test_dict['labels'][i]].append(img.tolist())
            else:
                p = []
                # print(type(list(img)))
                p.append(img.tolist())
                label_dict.update({test_dict['labels'][i]: p})
        return label_dict


# read_img(0)


def read_next_batch(start_position, batch_size):
    j = start_position // 10000 + 1
    k = start_position % 10000
    data_name = data_route + "/data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径
    data_dict = unpickle(data_name)
    pictures = []
    labels = []
    for picture in range(k, k+batch_size):
        point = []
        labels.append(data_dict['labels'][picture])
        for i in range(0, 1024):
            img = data_dict['data'][picture]  # ['data']为图片二进制数据
            R = img[i]
            G = img[1024+i]
            B = img[2048+i]
            point.append(np.array([R, G, B]))
            # print(point)
        point = np.array(point)
        # print(point)
        # point.reshape([32, 32, 3])
        pictures.append(point)
    pictures = np.array(pictures)
    pictures.reshape([batch_size, 32, 32, 3])
    # print(pictures)
    pictures = tf.convert_to_tensor(pictures)
    # print(pictures)
    # print(data_dict['data'][picture])
    # print(pictures)
    # print(type(pictures))
    labels = tf.convert_to_tensor(labels)
    # print(type(labels))
    return pictures, labels


# read_next_batch(4, 5)
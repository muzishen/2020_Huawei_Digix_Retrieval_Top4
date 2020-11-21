import pandas as pd
import numpy as np
import os

def filter_data(label_path, data_path, num):
    label = pd.read_csv(label_path, header=None)
    label[1] = label[0].map(lambda x: x.split('/')[0])
    label[2] = label[0].map(lambda x: x.split('/')[1])
    data_label = label[1].value_counts()[label[1].value_counts() < num]
    label_index = data_label.index.to_list()
    count = 0
    for i in label_index:
        image_name = label[0][label[1].values == i].values
        #print(image_name)
        for j in image_name:
            de_image_name = j.split('_')[1].replace("/", '_')
            de_image_path = os.path.join(data_path, de_image_name)
            os.remove(de_image_path)
            count = count + 1
    print(count)
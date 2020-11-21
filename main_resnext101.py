import os
import moxing as mox
from naie.context import Context

from naie.datasets import get_data_reference
from naie.feature_processing import data_flow

#下载训练数据集
data_reference = get_data_reference(dataset="train_data1", dataset_entity="train_data")  #数据集服务里面新建数据集的名字 例如：rap
file_paths = data_reference.get_files_paths()   #为了知道数据集地址
print(file_paths)
#将云端数据集移动到本地进行使用
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/384874b7ed094e1f90c42d874a8f13e5/Dataset/train_data1/train_data/data/trains.z01', 
'/cache/trains.z01')
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/384874b7ed094e1f90c42d874a8f13e5/Dataset/train_data1/train_data/data/trains.z02', 
'/cache/trains.z02')

data_reference = get_data_reference(dataset="train_data2", dataset_entity="train_data")  #数据集服务里面新建数据集的名字 例如：rap
file_paths = data_reference.get_files_paths()   #为了知道数据集地址
print(file_paths)
#将云端数据集移动到本地进行使用
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/384874b7ed094e1f90c42d874a8f13e5/Dataset/train_data2/train_data/data/trains.zip', 
'/cache/trains.zip')
#解压到你需要的地方
os.chdir('/cache/')
os.system('zip /cache/trains.zip -s=0 --out /cache/train.zip')
os.system('jar -xf /cache/train.zip')



# #pseduo images
# data_reference = get_data_reference(dataset="pseudo_data", dataset_entity="data")  #数据集服务里面新建数据集的名字 例如：rap
# file_paths = data_reference.get_files_paths()   #为了知道数据集地址
# print(file_paths)
# #将云端数据集移动到本地进行使用
# mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/384874b7ed094e1f90c42d874a8f13e5/Dataset/pseudo_data/data/data/pseudo_images.zip', 
# '/cache/pseudo_images.zip')
# os.system('jar -xf /cache/pseudo_images.zip')
# os.system('pwd')
# os.system('ls /cache')
# # os.system('exit(0)')

# #gallery数据集
# data_reference = get_data_reference(dataset="gallery", dataset_entity="gallery")  #数据集服务里面新建数据集的名字 例如：rap
# file_paths = data_reference.get_files_paths()   #为了知道数据集地址
# #将云端数据集移动到本地进行使用
# mox.file.copy('s3://bucket-db85kfn3/08c141ad8d00f3881f02c00cc0c8c1b3/7f8f4b22fe774e6e88d54099b2bf9abc/Dataset/gallery/gallery/data/gallery.zip', 
# '/cache/gallery.zip')
# data_reference = get_data_reference(dataset="query", dataset_entity="query")  #数据集服务里面新建数据集的名字 例如：rap
# file_paths = data_reference.get_files_paths()   #为了知道数据集地址
# mox.file.copy('s3://bucket-db85kfn3/08c141ad8d00f3881f02c00cc0c8c1b3/7f8f4b22fe774e6e88d54099b2bf9abc/Dataset/query/query/data/query.zip', 
# '/cache/query.zip')

os.chdir('/cache/')
# os.system('ls')
# os.system('jar -xf /cache/gallery.zip')
# os.system('jar -xf /cache/query.zip')

os.remove('trains.z01')
os.remove('trains.z02')
os.remove('trains.zip')
# os.remove('query.zip')
# os.remove('gallery.zip')
os.remove('train.zip')
os.system('pwd')
#os.system('mv ./pseudo_images_fileter3_0.5/*  ./train/')
os.system('ls')

# 下载预训练模型， 同上
#resnet50
data_reference = get_data_reference(dataset="pretrian_model", dataset_entity="resnext101_ibn")
file_paths = data_reference.get_files_paths()
print(file_paths)
mox.file.copy('s3://bucket-b7ix9vgp/06369383ad0010e41fc3c0169d929d78/384874b7ed094e1f90c42d874a8f13e5/Dataset/pretrian_model/resnext101_ibn/data/resnext101_ibn_a.pth', 
'/cache/resnext101_ibn_a.pth')

# data_reference = get_data_reference(dataset="train_label", dataset_entity="label_txt")
# file_paths = data_reference.get_files_paths()
# print(file_paths)
# mox.file.copy('s3://bucket-db85kfn3/08c141ad8d00f3881f02c00cc0c8c1b3/7f8f4b22fe774e6e88d54099b2bf9abc/Dataset/train_label/label_txt/data/label.txt', 
# '/cache/label.txt')


hw_model_path = Context.get_model_path()
print(hw_model_path)


#os.system('docker run --shm-size 1G')
os.chdir('/cache/user-job-dir/codes/code')
os.system('pwd')
os.system('ls')
os.system('python main.py')
os.system('zip -r ./log.zip  ./log/*')
mox.file.copy('/cache/user-job-dir/codes/code/log.zip', os.path.join(hw_model_path,  'log.zip'))






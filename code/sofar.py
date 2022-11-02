#import necessary packages
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#define unpickle data function, using on 3 files of cifar100, each return a dictionary
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#prepare directories for all files
root_cifar="/home/tkdc/chogit/code/data/cifar-100-python/"
root_train=root_cifar+"train"
root_test=root_cifar+"test"
root_meta=root_cifar+"meta"
#unpickle things
traindata=unpickle(root_train)
testdata=unpickle(root_test)
metadata=unpickle(root_meta)
#test - see type of train dataset, expect return <class dict> - true 
#print(type(traindata))

for keyss in traindata:
    print(keyss,type(traindata[keyss]))
#b'filenames' <class 'list'>
#b'batch_label' <class 'bytes'>
#b'fine_labels' <class 'list'> = class of images
#b'coarse_labels' <class 'list'> = superclass of images
#b'data' <class 'numpy.ndarray'>

for keyss in metadata:
    print(keyss,type(metadata[keyss]))
X_train=traindata[b'data']
#print(X_train.shape)
#(50000, 3072)
#reshape&transpose
X_train = X_train.reshape(len(X_train),3,32,32)
# Transpose the whole data
X_train = X_train.transpose(0,2,3,1)

#test to see the image: worked
#image = X_train[3]
#plt.imshow(image)
#plt.show() 
print(X_train.shape,type(X_train))

class_label=traindata[b'fine_labels']
superclass_label=traindata[b'coarse_labels']
class_name=metadata[b'fine_label_names']
#for i in range(len(class_name)):
#    print("i: ",i," for: ", class_name[i])
superclass_name=metadata[b'coarse_label_names']
#for i in range(len(superclass_name)):
#    print("i: ",i," for: ", superclass_name[i])
#i:  5  for:  b'household_electrical_devices'
#i:  6  for:  b'household_furniture'

def identify_superclass_labels_case(x):
    ret=[]
    for i in range(len(superclass_label)):
        if superclass_label[i]==x:
            ret.append(i)
    return ret

#def divide_class_labels_case(x,y,p):
#    return 
def merge_class_labels_case(x,y):
    return x+y

a=identify_superclass_labels_case(5)
b=identify_superclass_labels_case(6)
c=merge_class_labels_case(a,b)

#test:
#yo=655
#image=X_train[c[yo]]
#plt.imshow(image) 
#plt.title("Coarse Label Name:{} \n Fine Label Name:{}"
#          .format(superclass_name[superclass_label[c[yo]]], class_name[class_label[c[yo]]]))
#plt.show() 




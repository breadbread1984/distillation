# A distillation practice implemented with Caffe2
### Introduction

This project distills MobileID and MobileNet with the feature vectors got from ResNet.

### build everything

Build everything with the following command
```Shell
make
```

### create database of feature vectors output by ResNet

prepare all facial images in one directory. the images can be organized in any structure within the directory. Then strike the following command
```Shell
./createLMDB -i <facial image directory> -o dataset
```
### start distillation

distill into a MobileID network with
```Shell
./train_MobileID
```

distill into a MobileNet network with
```Shell
./train_MobileNet
```

### Test the effectiveness of the distilled network with ROC

calculate ROC with
```Shell
./roc -i <LFW root directory> -p <sample pair list>
```
The area under curve (AUC) is given at the end of the execution. You can also draw the ROC by the following Matlab command
```Shell
load roc.txt
plot(roc(:,1),roc(:,2))
```

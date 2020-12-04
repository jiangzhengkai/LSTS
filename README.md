# Learning Where to Focus for Efficient Video Object Detection

![image](./docs/figures/lsts.png)
[Paper](https://arxiv.org/pdf/1911.05253.pdf)
[Project Page](https://jiangzhengkai.github.io/LSTS/)


## Installation

1. Clone this repository. 

~~~
git clone https://github.com/jiangzhengkai/LSTS.git
~~~
2. Run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 62ecb60
	git submodule update
	```
	3.2 Copy operators in `lib/ops/*` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r lib/ops/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j4
	```
	3.4 Install the MXNet Python binding by
	```
	cd python
	sudo python setup.py install
	```

## Preparation for Training & Testing

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset, and make sure it looks like this:

	```
	./data/ILSVRC2015/
	./data/ILSVRC2015/Annotations/DET
	./data/ILSVRC2015/Annotations/VID
	./data/ILSVRC2015/Data/DET
	./data/ILSVRC2015/Data/VID
	./data/ILSVRC2015/ImageSets
	```

2. Please download ImageNet pre-trained ResNet-v1-101 model and Flying-Chairs pre-trained FlowNet model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMOBdCBiNaKbcjPrA) (for users from Mainland China, please try [Baidu Yun](https://pan.baidu.com/s/1nuPULnj)), and put it under folder `./model`. Make sure it looks like this:


	```
	./model/pretrained_model/resnet_v1_101-0000.params
	./model/pretrained_model/flownet-0000.params
	```

## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/lsts/cfgs`.
2. To perform experiments, run the python script with the corresponding config file as input.
    ```
    python experiments/lsts/lsts_end2end_train_test.py --cfg experiments/lsts_rfcn/cfgs/lsts_network_uniform.yaml
    ```


## Bibtex
```
@inproceedings{jiang2020learning,
  title={Learning Where to Focus for Efficient Video Object Detection},
  author={Jiang, Zhengkai and Liu, Yu and Yang, Ceyuan and Liu, Jihao and Gao, Peng and Zhang, Qian and Xiang, Shiming and Pan, Chunhong},
  booktitle={European Conference on Computer Vision},
  year={2020},
}
```




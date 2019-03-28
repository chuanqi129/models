# INT8 Calibration for Image Classification Models


## PaddlePaddle build and installation
>Requirements: cmake >= 3.0, python protobuf >= 3.0, patchelf
```shell
git clone -b calibration_tool https://github.com/chuanqi129/Paddle.git 
cd paddle
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./tmp -DWITH_GPU=OFF -DWITH_MKLDNN=ON -DWITH_TESTING=ON -DWITH_PROFILER=ON -DWITH_MKL=ON -DWITH_INFERENCE_API_TEST=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON 
make -j
make install -j
make -j fluid_lib_dist
pip install --force-reinstall --user ./tmp/opt/paddle/share/wheels/paddlepaddle-*.whl
```

## Data preparation
An example for ImageNet classification is as follows. First of all, preparation of imagenet data can be done as:
```shell
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```

In the shell script ```download_imagenet2012.sh```,  there are three steps to prepare data:

**step-1:** Register at ```image-net.org``` first in order to get a pair of ```Username``` and ```AccessKey```, which are used to download ImageNet data.

**step-2:** Download ImageNet-2012 dataset from website. The training and validation data will be downloaded into folder "train" and "val" respectively. Please note that the size of data is more than 40 GB, it will take much time to download. Users who have downloaded the ImageNet data can organize it into ```data/ILSVRC2012``` directly.

**step-3:** Download training and validation label files.

## Calibration from FP32 weights
**I. Download pre-trained FP32 weights from [`README`](./README.md) file as below [table](#supported-models).**

**II. Save pre-trained FP32 weights as a FP32 model by below commands.**

```shell
python save_model_from_var.py 
		--pretrained_model pretrained_model/ResNet50_pretrained/ 
		--model ResNet50 
		--saved_model new_weights/resnet50
```
**parameter introduction:**
* **model**: name model to use. Default: "SE_ResNeXt50_32x4d".

* **pretrained_model**: the pretrained weights dir. Default: None.

* **saved_model**: the saved model dir. Default: "new_model".

* **model_category**: the category of models, ("models"|"models_name"). Default: "models_name".

You also can save a **fake FP32 weights** for models by below command:
```shell
python save_model_from_var.py 
		--pretrained_model none \
		--model GoogleNet \
		--saved_model fake_weights/googlenet
```
**III. Calibration INT8 model**
Suppose you have downloaded the ImageNet Val dataset in `./data/ILSVRC2012/`.

```shell
FLAGS_use_mkldnn=true OMP_NUM_THREADS=28 python calibration.py 
		--model=ResNet50 \
		--batch_size=50 \
		--class_dim=1000 \
		--image_shape=3,224,224 \
		--with_mem_opt=True \
		--use_gpu=False \
		--first_conv_int8=True \
		--pretrained_model=new_weights/resnet50 \
		--out=new_weights_int8/resnet50_int8
```
For currently supported model, you just change `model`, `pretrained_model` and `out` according your models. And then you can get a INT8 models in `./new_weights_int8/resnet50_int8`.

## Python API evaluation

In this method, we just need use below command to run accuracy or performance test.
```shell
FLAGS_use_mkldnn=true python eval_tp_with_model.py
		--class_dim=1000 \
		--image_shape=3,224,224 \
		--with_mem_opt=True \
		--use_gpu=False  \
		--pretrained_model=new_weights_int8/resnet50_int8  \
		--skip_batch_num=0  \
		--use_transpiler=True  \
		--profiler=True  \
		--iteration=1000  \
		--batch_size=50
```
Per `pretrained_model` you can specify a **FP32** models or an **INT8** models. When you use this script to measure performance, we suggested you add `--use_fake_data=true` and `--skip_batch_num=10`.

## C API evaluation

>We prefer this method.

**I. Build CAPI infer image classfication application**

```shell
cd ./capi
mkdir build && cd build
cmake .. -DPADDLE_ROOT=/path/to/paddle/build/fluid_install_dir/
make
```
**II. Evaluation Models Accuracy and performance**
```shell
FLAGS_use_mkldnn=true ./build/infer_image_classification \
        --infer_model=new_weights_int8/resnet50_int8 \
        --batch_size=50 \
        --profile \
        --skip_batch_num=0 \
        --iterations=1000  \
        --data_list=../data/ILSVRC2012/val_list.txt \
        --data_dir=../data/ILSVRC2012/ \
        --use_mkldnn \
        --paddle_num_threads=1  \
        --use_fake_data=false
```
Per `infer_model` you also can specify a **FP32** models or an **INT8** models. When you want measure performance of the model, you can change the `use_fake_data` as `true`.

## Supported models

Models consists of two categories: Models with specified parameters names in model definition and Models without specified parameters, Generate named model by indicating ```model_category = models_name```.

Models are trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each pre-defined epoches, if not special introduced. Available top-1/top-5 validation accuracy on ImageNet 2012 are listed in table. Pretrained models can be downloaded by clicking related model names.


- Released models: specify parameter names

|model | top-1/top-5 accuracy(PIL)| top-1/top-5 accuracy(CV2) |
|- |:-: |:-:|
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.zip) | 56.71%/79.18% | 55.88%/78.65% |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.zip) | 69.22%/89.09% | 69.01%/88.90% |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.zip) | 70.14%/89.48% | 69.83%/89.13% |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.zip) | 72.08%/90.63% | 71.65%/90.57% |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.zip) | 72.56%/90.83% | 72.32%/90.98% |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.zip) | 70.91%/89.54% | 70.51%/89.35% |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.zip) | 71.90%/90.55% | 71.53%/90.41% |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.zip) | 76.35%/92.80% | 76.22%/92.92% |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.zip) | 77.49%/93.57% | 77.56%/93.64% |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.zip) | 78.12%/93.93% | 77.92%/93.87% |


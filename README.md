## HRDecoder: High-Resolution Decoder Network for Fundus Image Lesion Segmentation

Our code is implemented based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation).

### Environment

In this work, we used:
- python=3.8.5
- pytorch=2.0.1
- mmsegmentation=0.16.0
- mmcv=1.7.1

The environment can be installed with:
```
conda create -n hrdecoder python=3.8.5
conda activate hrdecoder

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -U openmim
mim install mmcv==1.7.1
```

### Datasets
We carried out experiments on two public datasets, i.e. IDRiD and DDR. For IDRiD, please download the segmentation part from [here](https://ieee-dataport.s3.amazonaws.com/open/3754/A.%20Segmentation.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20Segmentation.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240315%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240315T053048Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=de00f49e9a770569b25982825c74f0b74a69e40fb965de881a8efca993f5b71f). For DDR dataset, please download the segmentation part from [here](https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC).

Please first change the original structure of datasets and organize like this:
```
IDRiD(segmentation part)
|——images
|  |——train
|  |  |——IDRiD_01.jpg
|  |  |——...
|  |  |——IDRiD_54.jpg
|  |——test
|  |  |——...
|——labels
|  |——train
|  |  |——EX
|  |  |	 |——IDRiD_01.tiff
|  |  |	 |——...
|  |  |  |——IDRiD_54.tiff
|  |  |——HE
|  |  |	 |——...
|  |  |——SE
|  |  |	 |——...
|  |  |——MA
|  |  |  |——...
|  |——test
|  |  |——...
```

Please note that the original GT with `.tiff` format can not be directly used for training, so you can use the following command to convert the labels to `.png` format:
```
python tools/convert_dataset/idrid.py
python tools/convert_dataset/ddr.py
```
Then the structure of dataset will be converted to:
```
IDRiD(segmentation part)
|——images
|  |——train
|  |  |——IDRiD_01.jpg
|  |  |——IDRiD_02.jpg
|  |  |——...
|  |  |——IDRiD_54.jpg
|  |——test
|  |  |——...
|——labels
|  |——train
|  |  |——IDRiD_01.png
|  |  |——IDRiD_02.png
|  |  |——...
|  |  |——IDRiD_54.png
|  |——test
|  |  |——...
```

Finally the structure of the project should be like this:
```
HRDecoder
|——configs
|——mmseg
|——tools
|——data
|  |——IDRiD(segmentation part)
|  |  |——images
|  |  |  |——train
|  |  |  |——test
|  |  |——labels
|  |  |  |——train
|  |  |  |——test
|  |——DDR(segmentation part)
|  |  |——images
|  |  |  |——train
|  |  |  |——test
|  |  |——labels
|  |  |  |——train
|  |  |  |——test
```

### Training and Testing

To train or test a model using MMsegmentation framework, you can use commands like this:
```
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=18940 tools/dist_train.sh configs/lesion/hrdecoder_fcn_hr48_idrid_2880x1920-slide.py 4
or
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=18940 tools/dist_train.sh configs/lesion/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide.py 4

# test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=17940 tools/dist_test.sh configs/lesion/hrdecoder_fcn_hr48_idrid_2880x1920-slide.py work_dirs/hrdecoder_fcn_hr48_idrid_2880x1920-slide/latest.pth 4 --eval mIoU
or
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=17940 tools/dist_test.sh configs/lesion/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide.py work_dirs/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide/latest.pth 4 --eval mIoU
```

The trained model will be stored to `work_dirs/`.

In this project, we supply two version of implemention: HRDecoder and Efficient-HRDecoder.

HRDecoder is the version utilized for presenting the results in our paper. While Efficient-HRDecoder is an improved version of HRDecoder to further reduce computational overhead and memory usage. Specifically, we simply compress the dimension of the extracted features using a 1x1 convolutional layer right after the backbone. This helps save a lot of memory.





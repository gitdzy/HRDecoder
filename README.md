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
```

### Datasets

We carried out experiments on two public datasets, i.e. IDRiD and DDR. For IDRiD, please download the segmentation part from [here](https://ieee-dataport.s3.amazonaws.com/open/3754/A.%20Segmentation.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20Segmentation.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240315%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240315T053048Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=de00f49e9a770569b25982825c74f0b74a69e40fb965de881a8efca993f5b71f). For DDR dataset, please download the segmentation part from [here]().



### Training and Testing


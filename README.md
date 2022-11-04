# KE-RCNN

Official implementation of [**KE-RCNN**](https://arxiv.org/pdf/2206.10146.pdf) for part-level attribute parsing. It based on [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Installation
- pytorch 1.10.0 
- python 3.7.0
- [mmdet 2.25.1](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)
- [fashionpeida-API](https://github.com/KMnP/fashionpedia-api)
- einops

## Dataset
You need to download the datasets and annotations follwing this repo's formate

- [FashionPedia](https://github.com/cvdfoundation/fashionpedia)
- [knowledge_matrix](https://drive.google.com/file/d/1m1ycDqK6wvdvlLwz7jyyAuIGjyhdggBe/view?usp=sharing)

Make sure to put the files as the following structure:

```
  ├─data
  │  ├─fashionpedia
  │  │  ├─train
  │  │  ├─test
  │  │  │─instances_attribute_train2020.json
  │  │  │─instances_attribute_val2020.json
  |  |  |─train_norm_attr_knowledge_matrix.npy
  |
  ├─work_dirs
  |  ├─ke_rcnn_r50_fpn_fashion_1x
  |  |  ├─epoch32.pth
  ```

## Results and Models

### FashionPedia

|  Backbone    |  LR  | AP_iou+f1 | AP_mask_iou+f1 | DOWNLOAD |
|--------------|:----:|:---------:|:--------------:|:--------:|
|  R-50        |  1x  | 39.6      | 36.4           |[model](https://drive.google.com/file/d/10mz200uBm-2DqAYN9pbSTYG4sk8SNou8/view?usp=sharing)|
|  R-101       |  1x  | 39.9      | 36.6           |[model](https://drive.google.com/file/d/1Z1gBiFcL1YuVpL3jmYvs-uqylW8noROW/view?usp=sharing)|
|  HRNet-w18   |  1x  | 38.0      | 35.3           |[model](https://drive.google.com/file/d/1fEkR3ylrw5_sRU4ofaq0URCJuCrBb-AJ/view?usp=sharing)|
|  Swin-tiny   |  1x  | 43.7      | 40.5           |[model](https://drive.google.com/file/d/1sirL9xbeASFi3Ey1TpjUsXSQIGQstZP3/view?usp=sharing)|

- This is a reimplementation. Thus, the numbers are slightly different from our original paper.
## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/ke_rcnn/ke_rcnn_r50_fpn_fashion_1x.py work_dirs/ke_rcnn_r50_fpn_fashion_1x/epoch32.pth 8 --format-only --eval-options "jsonfile_prefix=work_dirs/ke_rcnn_r50_fpn_fashion_1x/ke_rcnn_r50_fpn_fashion_1x_val_result"

# eval, noted that should change the json path produce by previous step.
python eval/fashion_eval.py
```

## Training
```
# training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/ke_rcnn/ke_rcnn_r50_fpn_fashion_1x.py 8
```

## Citation
```
@article{wang2022ke,
  title={KE-RCNN: unifying knowledge based reasoning into part-level attribute parsing},
  author={Wang, Xuanhan and Song, Jingkuan and Chen, Xiaojia and Cheng, Lechao and Gao, Lianli and Shen, Heng Tao},
  journal={arXiv preprint arXiv:2206.10146},
  year={2022}
}
```

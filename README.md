# Model with Spatially Weighted Pooling (SWP)

An imeplementation of `Model+SWP` using Keras (TF2) framework. Models supported: ResNet{50, 101}, VGG16, AlexNet

## Implementation
### ResNet_SWP
- `base_model_trainable`: bool = True
- `input_shape`:(224, 224, 3)
- `swp_num_of_masks`: 9
- `fc_nodes`: 1024
- `weight_decay`: 0.0005
- `stddev`: 0.005
- `dropout_ratio`: 0.5

You should set `learning_rate` of optimizers in `{0.001, 0.0001, 0.00001}`

### VGG16_SWP
- `base_model_trainable`: bool = True
- `input_shape`:(224, 224, 3)
- `swp_num_of_masks`: 9
- `fc_nodes`: 512
- `weight_decay`: 0.0005
- `stddev`: 0.005
- `dropout_ratio`: 0.5

You should set `learning_rate` of optimizers in `{0.001, 0.0001, 0.00001}`

### AlexNet_SWP
- `base_model_trainable`: bool = True
- `input_shape`:(227, 227, 3)
- `swp_num_of_masks`: 9
- `fc_nodes`: 512
- `weight_decay`: 0.0005
- `stddev`: 0.005
- `dropout_ratio`: 0.5

You should set `learning_rate` of optimizers in `{0.001, 0.0001}`

## Pretrained model for ResNet50_SWP
I trained the CompCars dataset on ResNet50_SWP. To use it, use this command to combine files:
```
cat pretrained/resnet50_* > resnet50.h5
```

## Citation
The idea is from two these papers (special thanks to the authors):
```
[https://ieeexplore.ieee.org/document/7891907]
Q. Hu, H. Wang, T. Li and C. Shen, "Deep CNNs With Spatially Weighted Pooling for Fine-Grained Car Recognition," in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 11, pp. 3147-3156, Nov. 2017, doi: 10.1109/TITS.2017.2679114.
```
```
[https://arxiv.org/abs/1506.08959]
Q. Hu, H. Wang, T. Li and C. Shen, "Deep CNNs With Spatially Weighted Pooling for Fine-Grained Car Recognition," in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 11, pp. 3147-3156, Nov. 2017, doi: 10.1109/TITS.2017.2679114.
```
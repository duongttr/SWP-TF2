"""
This code is implemented from these of two papers:

- Q. Hu, H. Wang, T. Li and C. Shen, "Deep CNNs With Spatially Weighted Pooling for Fine-Grained Car Recognition," 
in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 11, pp. 3147-3156, Nov. 2017, doi: 10.1109/TITS.2017.2679114.

- Yang, L., Luo, P., Loy, C. C., & Tang, X. (2015). A large-scale car dataset for fine-grained categorization and verification. 
2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2015.7299023
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, VGG16
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import random_normal
from SWP import SWPLayer

class ResNet_SWP(tf.keras.Model):
    def __init__(self,
               base_model: str = 'resnet50',
               base_model_trainable: bool = True,
               input_shape: tuple = (224, 224, 3),
               num_classes: int = 431,
               swp_num_of_masks: int = 9,
               fc_nodes: int = 1024,
               weight_decay: float = 0.0005,
               stddev: float = 0.005,
               dropout_ratio: float = 0.5,
               random_state: int =None
               ):
        """
        # ResNet_SWP
        `base_model` (str): resnet50 or resnet101
        `base_model_trainable` (bool): set `base_model` to `trainable`
        `input_shape` (tuple): input shape of images
        `num_classes` (int): number of classes
        `swp_num_of_masks` (int): number of masks of SWP layer
        `fc_nodes` (int): number of nodes of fully-connected layers
        `weight_decay` (float): L2 regularized multiplier
        `stddev` (float): standard deviation of random normal
        `dropout_ratio` (float): ratio of Dropout layer to prevent overfitting problem
        `random_state` (int): set random state for random initialized parameters.
        """
        super(ResNet_SWP, self).__init__()

        # initialize layers
        if base_model == 'resnet50':
            self.base_model = ResNet50(include_top=False, input_shape=input_shape)
        else:
            self.base_model = ResNet101(include_top=False, input_shape=input_shape)
        
        self.base_model.trainable = base_model_trainable

        self.swp_layer = SWPLayer(swp_num_of_masks, seed=random_state)
        self.batch_norm_layer_1 = BatchNormalization()
        self.batch_norm_layer_2 = BatchNormalization()
        self.relu_act_layer_1 = Activation('relu')
        self.relu_act_layer_2 = Activation('relu')
        self.flatten_layer = Flatten()
        self.fc_layer = Dense(fc_nodes, kernel_regularizer=L2(weight_decay),
                            kernel_initializer=random_normal(stddev=stddev, seed=random_state))
        self.dropout_layer = Dropout(dropout_ratio, seed=random_state)
        self.output_layer = Dense(num_classes,
                            kernel_regularizer=L2(weight_decay),
                            activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.swp_layer(x)
        x = self.batch_norm_layer_1(x)
        x = self.flatten_layer(x)
        x = self.relu_act_layer_1(x)
        x = self.fc_layer(x)
        x = self.batch_norm_layer_2(x)
        x = self.relu_act_layer_2(x)
        x = self.dropout_layer(x)
        return self.output_layer(x)


class AlexNet_SWP(tf.keras.Model):
    def __init__(self, 
                base_model_trainable: bool = True,
                input_shape: tuple = (227, 227, 3),
                num_classes: int = 431,
                swp_num_of_masks: int = 9,
                fc_nodes: int = 512,
                weight_decay: float = 0.0005,
                stddev: float = 0.005,
                dropout_ratio: float = 0.5,
                random_state: int =None):
        """
        # AlexNet_SWP
        `base_model_trainable` (bool): set `base_model` to `trainable`
        `input_shape` (tuple): input shape of images
        `num_classes` (int): number of classes
        `swp_num_of_masks` (int): number of masks of SWP layer
        `fc_nodes` (int): number of nodes of fully-connected layers
        `weight_decay` (float): L2 regularized multiplier
        `stddev` (float): standard deviation of random normal
        `dropout_ratio` (float): ratio of Dropout layer to prevent overfitting problem
        `random_state` (int): set random state for random initialized parameters.
        """
        super(AlexNet_SWP, self).__init__()

        # Initialize model
        # 1st Convolutional Layer
        self.conv2d_1 = Conv2D(filters=96, input_shape=input_shape, 
                                kernel_size=(11,11), strides=(4,4), 
                                padding='valid')
        self.conv2d_1.trainable = base_model_trainable
        self.act_1 = Activation('relu')
        
        # Max Pooling
        self.max_pooling_2d_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')

        # 2nd Convolutional Layer
        self.conv2d_2 = Conv2D(filters=256, kernel_size=(11,11), 
                            strides=(1,1), padding='valid')
        self.conv2d_2.trainable = base_model_trainable
        self.act_2 = Activation('relu')
        # Max Pooling
        self.max_pooling_2d_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')

        # 3rd Convolutional Layer
        self.conv2d_3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid')
        self.conv2d_3.trainable = base_model_trainable
        self.act_3 = Activation('relu')

        # 4th Convolutional Layer
        self.conv2d_4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid')
        self.conv2d_4.trainable = base_model_trainable
        self.act_4 = Activation('relu')

        # 5th Convolutional Layer
        self.conv2d_5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid')
        self.conv2d_5.trainable = base_model_trainable
        self.act_5 = Activation('relu')
        # Max Pooling
        self.max_pooling_2d_5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')

        self.swp_layer = SWPLayer(swp_num_of_masks, seed=random_state)
        self.batch_norm_layer_1 = BatchNormalization()
        self.act_6 = Activation('relu')

        # Passing it to a Fully Connected layer
        self.flatten = Flatten()
        # 1st Fully Connected Layer
        self.fc_1 = Dense(fc_nodes,
                            kernel_regularizer=L2(weight_decay),
                            kernel_initializer=random_normal(stddev=stddev, seed=random_state))
        self.batch_norm_fc_1 = BatchNormalization()
        self.act_fc_1 = Activation('relu')
        # Add Dropout to prevent overfitting
        self.dropout_1 = Dropout(dropout_ratio, seed=random_state)

        # 2nd Fully Connected Layer
        self.fc_2 = Dense(fc_nodes, 
                            kernel_regularizer=L2(weight_decay),
                            kernel_initializer=random_normal(stddev=stddev, seed=random_state))

        self.batch_norm_fc_2 = BatchNormalization()
        self.act_fc_2 = Activation('relu')
        # Add Dropout
        self.dropout_2 = Dropout(dropout_ratio, seed=random_state)

        # Output Layer
        self.output_layer = Dense(num_classes, 
                            kernel_regularizer=L2(weight_decay),
                            activation='softmax')
    
    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.act_1(x)
        x = self.max_pooling_2d_1(x)
        x = self.conv2d_2(x)
        x = self.act_2(x)
        x = self.max_pooling_2d_2(x)
        x = self.conv2d_3(x)
        x = self.act_3(x)
        x = self.conv2d_4(x)
        x = self.act_4(x)
        x = self.conv2d_5(x)
        x = self.act_5(x)
        x = self.max_pooling_2d_5(x)
        x = self.swp_layer(x)
        x = self.batch_norm_layer_1(x)
        x = self.flatten(x)
        x = self.act_6(x)
        x = self.fc_1(x)
        x = self.batch_norm_fc_1(x)
        x = self.act_fc_1(x)
        x = self.dropout_1(x)
        x = self.fc_2(x)
        x = self.batch_norm_fc_2(x)
        x = self.act_fc_2(x)
        x = self.dropout_2(x)
        return self.output_layer(x)

        
class VGG16_SWP(tf.keras.Model):
    def __init__(self, 
                base_model_trainable: bool = True,
                input_shape: tuple = (224, 224, 3),
                num_classes: int = 431,
                swp_num_of_masks: int = 9,
                fc_nodes: int = 512,
                weight_decay: float = 0.0005,
                stddev: float = 0.005,
                dropout_ratio: float = 0.5,
                random_state: int =None):
        """
        # VGG16_SWP
        `base_model_trainable` (bool): set `base_model` to `trainable`
        `input_shape` (tuple): input shape of images
        `num_classes` (int): number of classes
        `swp_num_of_masks` (int): number of masks of SWP layer
        `fc_nodes` (int): number of nodes of fully-connected layers
        `weight_decay` (float): L2 regularized multiplier
        `stddev` (float): standard deviation of random normal
        `dropout_ratio` (float): ratio of Dropout layer to prevent overfitting problem
        `random_state` (int): set random state for random initialized parameters.
        """
        super(VGG16_SWP, self).__init__()

        # Initilize model

        self.base_model = VGG16(include_top=False, input_shape=input_shape)
        self.base_model.trainable = base_model_trainable

        self.fc_1 = Dense(fc_nodes, kernel_regularizer=L2(weight_decay),
                          kernel_initializer=random_normal(stddev=stddev, seed=random_state))
            
        self.fc_2 = Dense(fc_nodes, kernel_regularizer=L2(weight_decay),
                          kernel_initializer=random_normal(stddev=stddev, seed=random_state))

        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()

        self.dropout_1 = Dropout(dropout_ratio)
        self.dropout_2 = Dropout(dropout_ratio)

        self.act_1 = Activation('relu')
        self.act_2 = Activation('relu')
        self.act_3 = Activation('relu')
        
        self.swp = SWPLayer(K=swp_num_of_masks, seed=random_state)

        self.flatten = Flatten()

        self.output_layer = Dense(num_classes, activation='softmax',
                            kernel_regularizer=L2(weight_decay))
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.swp(x)
        x = self.batch_norm_1(x)
        x = self.flatten(x)
        x = self.act_1(x)
        x = self.fc_1(x)
        x = self.batch_norm_2(x)
        x = self.act_2(x)
        x = self.dropout_1(x)
        x = self.fc_2(x)
        x = self.batch_norm_3(x)
        x = self.act_3(x)
        x = self.dropout_2(x)
        return self.output_layer(x)
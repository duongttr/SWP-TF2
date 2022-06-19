"""
This code is implemented from these of two papers:

- Q. Hu, H. Wang, T. Li and C. Shen, "Deep CNNs With Spatially Weighted Pooling for Fine-Grained Car Recognition," 
in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 11, pp. 3147-3156, Nov. 2017, doi: 10.1109/TITS.2017.2679114.

- Yang, L., Luo, P., Loy, C. C., & Tang, X. (2015). A large-scale car dataset for fine-grained categorization and verification. 
2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2015.7299023
"""

import tensorflow as tf

class SWPLayer(tf.keras.layers.Layer):
    def __init__(self, K: int = 9, stddev: float = 0.005, seed: int = None, name: str = None):
        """
          K: Number of masks
          stddev: standard deviation of random normal of weights
          name: custom name for SWP Layer
        """
        super(SWPLayer, self).__init__(name=name)
        self.K = K
        self.stddev = stddev
        self.seed = seed

    def get_config(self):
        return {
            'K': self.K,
            'stddev': self.stddev,
            'seed': self.seed
        }

    def build(self, input_shape):
      super(SWPLayer, self).build(input_shape)

      weight_init = tf.keras.initializers.random_normal(stddev=self.stddev, seed=self.seed)
      mask_shape = (input_shape[1], input_shape[2], self.K)

      self.masks = tf.Variable(initial_value=weight_init(
          shape=(mask_shape),
          dtype='float32'
      ), trainable=True)
    
    def call(self, inputs):
      return tf.einsum('bhwc,hwm->bmc', inputs, self.masks)
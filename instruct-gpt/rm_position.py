import tensorflow as tf

class RmPosition(tf.keras.layers.Layer):
    """
    Position encoding via adding the trainable position bias.
    """

    def __init__(self, **kwargs):
        super(RmPosition, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is [Batch, Seq, Channel]
        seq = input_shape[1]
        hidden = input_shape[2]
        self.position_weight = self.add_weight(
            shape=(1, seq, hidden), 
            initializer='uniform',
            trainable=True,
            name='w_p')

    def call(self, inputs):
        return tf.add(inputs, self.position_weight)
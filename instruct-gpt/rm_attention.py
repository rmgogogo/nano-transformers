import tensorflow as tf

class RmMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention.
    """

    def __init__(self, head, residual_init_factor, **kwargs):
        """
        residual_init_factor is used to set the stddev of the initial weights, the weights is used to project the residual.
        As for why we do this, guess it's used to avoid too big residual in early phase.
        """
        super(RmMultiHeadAttention, self).__init__(**kwargs)
        self.head = head
        self.residual_init_factor = residual_init_factor

    def build(self, input_shape):
        # input_shape is [Batch, Seq, Channel]
        hidden = input_shape[0][2]
        # Weights for inputs. 
        #   stddev is bigger  => weights are more random  => then initial diff are more small  => then init attention-weights are more close
        #   It's possible we can have two different hidden, one is input, another is output.
        self.w_q = self.add_weight(
            shape=(hidden, hidden), 
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_q')
        self.w_k = self.add_weight(
            shape=(hidden, hidden), 
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_k')
        self.w_v = self.add_weight(
            shape=(hidden, hidden), 
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_v')
        self.w_m = self.add_weight(
            shape=(hidden, hidden), 
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2*self.residual_init_factor), trainable=True, name='w_m')
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "head": self.head,
            "residual_init_factor": self.residual_init_factor,
        })
        return config
    
    def call(self, inputs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        
        emb_q = tf.matmul(q, self.w_q)
        emb_k = tf.matmul(k, self.w_k)
        emb_v = tf.matmul(v, self.w_v)
        
        # [Batch, Seq, Channel] => [Head, Batch, Seq, Channel]
        multi_q = tf.stack(tf.split(emb_q, num_or_size_splits=self.head, axis=-1), axis=0)
        multi_k = tf.stack(tf.split(emb_k, num_or_size_splits=self.head, axis=-1), axis=0)
        multi_v = tf.stack(tf.split(emb_v, num_or_size_splits=self.head, axis=-1), axis=0)
        
        # Scale based on one head's shape, not all heads
        scale = tf.cast(multi_q.shape[-1] ** 0.5, tf.float32)
        dot_match = tf.matmul(multi_q, multi_k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(dot_match)

        # Sequence Mask or called Causal Mask, don't let model know future sequence
        # Another implementation is to set -INF, and then softmax
        attention_weights = tf.linalg.band_part(attention_weights, -1, 0)
        attention_weights = tf.math.divide(attention_weights, tf.reduce_sum(attention_weights, axis=3, keepdims=True))

        # Dropout
        # attention_weights = tf.keras.layers.Dropout(rate=dropout_rate)(attention_weights)
        
        # Convert from multiple style back to single style
        # [Head, Batch, Seq, Channel] => [Batch, Seq, Channel]
        weighted_v = tf.matmul(attention_weights, multi_v)
        weighted_v = tf.split(weighted_v, num_or_size_splits=self.head, axis=0)
        weighted_v = tf.concat(weighted_v, axis=-1)
        weighted_v = tf.squeeze(weighted_v, axis=0)

        # Mix the concated multi-head channels
        weighted_v = tf.matmul(weighted_v, self.w_m)

        # Dropout
        # weighted_v = tf.keras.layers.Dropout(rate=dropout_rate)(weighted_v)
        return weighted_v
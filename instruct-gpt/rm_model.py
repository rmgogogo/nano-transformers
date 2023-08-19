import tensorflow as tf
import math
from PIL import Image

from rm_position import RmPosition
from rm_attention import RmMultiHeadAttention

def plot_model(model, file_name = '/tmp/model.png'):
    """
    Plot the model
    """
    tf.keras.utils.plot_model(model, to_file=file_name, show_shapes=True, show_layer_activations=True)
    image = Image.open(file_name)
    image.show()

def get_model(max_len, hidden, head, vocab_size, dropout_rate, n_blocks):
    """
    Get the model
    - Input: (Batch, Seq)
    """
    # Encode (max_len)
    input = tf.keras.Input(shape=(max_len), name='input')
    data = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden, name='input_embedding')(input)
    data = RmPosition(name='input_positioning')(data)

    # Dropout
    data = tf.keras.layers.Dropout(rate=dropout_rate)(data)

    # Each block has two residual, one is attention, another is FF, so 2*n_blocks are the total residuals.
    residual_init_factor = 1/math.sqrt(2 * n_blocks)

    for i in range(n_blocks):
        # Norm, Attention and Add
        res =  tf.keras.layers.LayerNormalization(axis=-1, name=f'decode_attention_norm-{i}')(data)
        res = RmMultiHeadAttention(head=head, residual_init_factor=residual_init_factor, name=f'decode_attention-{i}')(inputs=(res, res, res))
        data = tf.keras.layers.Add(name=f'decode_attention_add-{i}')([data, res])
        
        # Norm, FeedForward, Dropout and Add
        res = tf.keras.layers.LayerNormalization(axis=-1, name=f'decode_ff_norm-{i}')(data)
        res = tf.keras.layers.Dense(units=hidden*2, activation='gelu', name=f'decode_ff_increse-{i}')(res)
        res = tf.keras.layers.Dense(units=hidden, activation=None, 
                                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2*residual_init_factor),
                                    name=f'decode_ff_decrease-{i}')(res)
        res = tf.keras.layers.Dropout(rate=dropout_rate)(res)
        data = tf.keras.layers.Add(name=f'decode_ff_add-{i}')([data, res])

    # Final norm
    data = tf.keras.layers.LayerNormalization(axis=-1, name='final_norm')(data)

    # Output (logits, not softmax, loss-fn side will take care it.)
    output = tf.keras.layers.Dense(vocab_size, activation=None, name='output')(data)

    model = tf.keras.Model(inputs=input, outputs=output, name='model')
    return model

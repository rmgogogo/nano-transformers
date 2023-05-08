import tensorflow as tf
import numpy as np
import codecs
import regex
import json

class TranslationData(object):
    """
    Translation DataSet, sentence pairs between DE and EN.
    Data is from https://github.com/greentfrapp/attention-primer/tree/master/5_translation/data
    """
    def __init__(self):
        self.en_file = "data/translation/train.tags.de-en.en"
        self.de_file = "data/translation/train.tags.de-en.de"
        self.en_samples = self._get_lines(self.en_file)
        self.de_samples = self._get_lines(self.de_file)
        self.n_samples = len(self.en_samples)
        self.en_dict = json.load(open("data/translation/en_dict.json", 'r', encoding='utf-8'))
        self.de_dict = json.load(open("data/translation/de_dict.json", 'r', encoding='utf-8'))
        self.en_vocab_size = len(self.en_dict)
        self.de_vocab_size = len(self.de_dict)

    def _get_lines(self, file):
        text = codecs.open(file, 'r', 'utf-8').read().lower()
        text = regex.sub("<.*>.*</.*>\r\n", "", text)
        text = regex.sub("[^\n\s\p{Latin}']", "", text)
        samples = text.split('\n')
        return samples

    def one_hot(self, sample, dictionary, max_len=20, sos=False, eos=False):
        sample = sample.split()[:max_len]
        while len(sample) < max_len:
            sample.append('<PAD>')
        if sos:
            tokens = ['<START>']
        else:
            tokens = []
        tokens.extend(sample)
        if eos:
            # TODO: <END>?
            tokens.append('<PAD>')
        
        idxs = []
        for token in tokens:
            try:
                idxs.append(dictionary.index(token))
            except:
                idxs.append(dictionary.index('<UNK>'))
        idxs = np.array(idxs)
        return np.eye(len(dictionary))[idxs]

    def prettify(self, sample, dictionary):
        idxs = np.argmax(sample, axis=1)
        return " ".join(np.array(dictionary)[idxs])
    
    def generate(self, max_len=20):
        idx = -1
        while True:
            idx = (idx + 1) % self.n_samples
            de = self.one_hot(self.de_samples[idx], self.de_dict, max_len=max_len, sos=False, eos=False)
            en_in = self.one_hot(self.en_samples[idx], self.en_dict, max_len=max_len, sos=True, eos=False)
            en_out = self.one_hot(self.en_samples[idx], self.en_dict, max_len=max_len, sos=False, eos=True)
            yield (de, en_in), en_out
    
    def get_generator(self, max_len):
        def generator():
            return self.generate(max_len=max_len)
        return generator

##################################################################################################################################

class RmPosition(tf.keras.layers.Layer):
    """
    Position encoding via adding the trainable weights.
    """

    def __init__(self, seq, hidden, **kwargs):
        super(RmPosition, self).__init__(**kwargs)
        self.position_weight = self.add_weight(shape=(1, seq, hidden), initializer='uniform', trainable=True, name='w_p')

    def get_config(self):
        """
        Required by Model Saving.
        """
        config = super().get_config()
        config.update({
            "seq": self.position_weight.shape[1],
            "hidden": self.position_weight.shape[2],
        })
        return config

    def call(self, inputs):
        return tf.add(inputs, self.position_weight)

##################################################################################################################################

class RmMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention.
    """

    def __init__(self, head, hidden, sequence_mask=False, **kwargs):
        super(RmMultiHeadAttention, self).__init__(**kwargs)
        self.head = head
        self.hidden = hidden
        self.sequence_mask = sequence_mask
        self.chunk_size = int(hidden / head)

        # Weights for inputs. 
        #   stddev is bigger  => weights are more random  => then initial diff are more small  => then init attention-weights are more close
        #   It's possible we can have two different hidden, one is input, another is output.
        self.w_q = self.add_weight(shape=(hidden, hidden), initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_q')
        self.w_k = self.add_weight(shape=(hidden, hidden), initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_k')
        self.w_v = self.add_weight(shape=(hidden, hidden), initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2), trainable=True, name='w_v')
    
    def get_config(self):
        """
        Required by Model Saving.
        """
        config = super().get_config()
        config.update({
            "head": self.head,
            "hidden": self.hidden,
            "sequence_mask": self.sequence_mask,
        })
        return config
    
    def call(self, inputs):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]
        
        emb_q = tf.matmul(q, self.w_q)
        emb_k = tf.matmul(k, self.w_k)
        emb_v = tf.matmul(v, self.w_v)
        
        multi_q = tf.stack(tf.split(emb_q, num_or_size_splits=self.head, axis=-1), axis=0)
        multi_k = tf.stack(tf.split(emb_k, num_or_size_splits=self.head, axis=-1), axis=0)
        multi_v = tf.stack(tf.split(emb_v, num_or_size_splits=self.head, axis=-1), axis=0)
        
        # Scale based on one head's shape, not all heads
        scale = tf.cast(multi_q.shape[-1] ** 0.5, tf.float32)
        dot_match = tf.matmul(multi_q, multi_k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(dot_match)

        # Sequence Mask (don't let model know future sequence)
        #   https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%887%EF%BC%89Mask%E6%9C%BA%E5%88%B6/
        if self.sequence_mask:
            attention_weights = tf.linalg.band_part(attention_weights, -1, 0)
            attention_weights = tf.math.divide(attention_weights, tf.reduce_sum(attention_weights, axis=3, keepdims=True))
        
        # Convert from multiple style back to single style
        weighted_v = tf.matmul(attention_weights, multi_v)
        weighted_v = tf.split(weighted_v, num_or_size_splits=self.head, axis=0)
        weighted_v = tf.concat(weighted_v, axis=-1)
        weighted_v = tf.squeeze(weighted_v, axis=0)
        
        return weighted_v

##################################################################################################################################

def get_model(max_len=20, hidden=64, head=4, en_vocab_size=1004, de_vocab_size=14966):
    """
    Get the model
    """
    # Encode (max_len, vocab_size)
    encodeInput = tf.keras.Input(shape=(max_len, de_vocab_size), name='encode_input')
    encodeEmb = tf.keras.layers.Dense(units=hidden, activation=None, name='encode_input_embedding')(encodeInput)
    encodeEmb = RmPosition(max_len, hidden, name='encode_input_positioning')(encodeEmb)

    #===== Transformer Encode Block Starts =====
    # Encode Attention Add Norm
    weighted_v = RmMultiHeadAttention(head=head, hidden=hidden, name='encode_attention')((encodeEmb, encodeEmb, encodeEmb))
    encodeEmb = tf.keras.layers.Add(name='encode_attention_add')([encodeEmb, weighted_v])
    encodeEmb =  tf.keras.layers.LayerNormalization(axis=-1, name='encode_attention_norm')(encodeEmb)
    
    # Encode FeedForward Add Norm
    encodeIncreased = tf.keras.layers.Dense(units=hidden*2, activation='relu', name='encode_ff_increse')(encodeEmb)
    encodeDecreased = tf.keras.layers.Dense(units=hidden, activation=None, name='encode_ff_decrease')(encodeIncreased)
    encodeEmb = tf.keras.layers.Add(name='encode_ff_add')([encodeEmb, encodeDecreased])
    encodeEmb = tf.keras.layers.LayerNormalization(axis=-1, name='encode_ff_norm')(encodeEmb)
    #===== Transformer Encode Block Ends =====
    
    # Decode (max_len+1, vocab_size), input has one more '<START>', use Mask;
    decodeInput = tf.keras.Input(shape=(max_len+1, en_vocab_size), name='decode_input')
    decodeEmb = tf.keras.layers.Dense(units=hidden, activation=None, name='decode_input_embedding')(decodeInput)
    decodeEmb = RmPosition(max_len+1, hidden, name='decode_input_positioning')(decodeEmb)
    
    #===== Transformer Decode Block Starts =====
    # Decode Attention Add Norm
    weighted_v = RmMultiHeadAttention(head=head, hidden=hidden, sequence_mask=True, name='decode_attention')((decodeEmb, decodeEmb, decodeEmb))
    decodeEmb = tf.keras.layers.Add(name='decode_attention_add')([decodeEmb, weighted_v])
    decodeEmb =  tf.keras.layers.LayerNormalization(axis=-1, name='decode_attention_norm')(decodeEmb)
    
    # Decode-Encode Attention Add Norm
    weighted_v = RmMultiHeadAttention(head=head, hidden=hidden, name='decode_encode_attention')((decodeEmb, encodeEmb, encodeEmb))
    decodeEmb = tf.keras.layers.Add(name='decode_encode_attention_add')([decodeEmb, weighted_v])
    decodeEmb =  tf.keras.layers.LayerNormalization(axis=-1, name='decode_encode_attention_norm')(decodeEmb)

    # Decode FeedForward Add Norm
    decodeIncreased = tf.keras.layers.Dense(units=hidden*2, activation='relu', name='decode_ff_increse')(decodeEmb)
    decodeDecreased = tf.keras.layers.Dense(units=hidden, activation=None, name='decode_ff_decrease')(decodeIncreased)
    decodeEmb = tf.keras.layers.Add(name='decode_ff_add')([decodeEmb, decodeDecreased])
    decodeEmb = tf.keras.layers.LayerNormalization(axis=-1, name='decode_ff_norm')(decodeEmb)
    #===== Transformer Decode Block Ends =====

    # Output
    output = tf.keras.layers.Dense(en_vocab_size, activation='softmax', name='output')(decodeEmb)

    model = tf.keras.Model(inputs=[encodeInput, decodeInput], outputs=output, name='model')
    return model

def plot_model(model):
    """
    Plot the model
    """
    from PIL import Image

    file_name = 'model.png'
    tf.keras.utils.plot_model(model, to_file=file_name, show_shapes=True, show_layer_activations=True)
    image = Image.open(file_name)
    image.show()

##################################################################################################################################

class BatchLoggingModel(tf.keras.Model):
    def __init__(self, model, tb_dir):
        super().__init__()
        self.model = model
        self.train_writer = tf.summary.create_file_writer(tb_dir)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('batch_loss', loss)
        return self.compute_metrics(x, y, y_pred, None)
    
    def call(self, x):
        x = self.model(x)
        return x

##################################################################################################################################

import os
import datetime

def train(max_len=20, hidden=64, head=4, batchsize=64, epochs=1, steps_per_epoch=None, tensorboard=False, tb_dir='logs', model_dir='saved_model/transformer_onehot'):
    """
    Train the model
    """
    translation_data = TranslationData()
    dataset = tf.data.Dataset.from_generator(
        translation_data.get_generator(max_len),
        output_signature=(((tf.TensorSpec(shape=(20, 14966), dtype=tf.int64), tf.TensorSpec(shape=(21, 1004), dtype=tf.int64)),
                           tf.TensorSpec(shape=(21, 1004), dtype=tf.int64))))
    dataset = dataset.prefetch(buffer_size=batchsize*1000).shuffle(buffer_size=batchsize*100).batch(batchsize)
    
    model = get_model(max_len=max_len, hidden=hidden, head=head,
                      en_vocab_size=translation_data.en_vocab_size, de_vocab_size=translation_data.de_vocab_size)
    if tensorboard:
        tb_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        model = BatchLoggingModel(model, tb_dir)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # metrics=["categorical_accuracy"],
    )

    print(f'Trainable params: {len(model.trainable_weights)}')
    print(f'Launch TensorBoard to check the logs:\n  tensorboard --logdir {tb_dir}')

    callbacks = []
    if tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
    steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else translation_data.n_samples // batchsize
    _ = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks) 
    model.model.save(model_dir)

def predict(input_string, max_len=20, model_dir='saved_model/transformer_onehot'):
    model=tf.keras.models.load_model(model_dir)
    translation_data = TranslationData()

    input_de = np.array([translation_data.one_hot(input_string, translation_data.de_dict, max_len=max_len)])
    output = ""
    for i in range(max_len):
        last_output = np.array([translation_data.one_hot(output, translation_data.en_dict, max_len=max_len, sos=True)])
        predicted = model.predict((input_de, last_output), verbose = 2)
        predicted_str = translation_data.prettify(predicted[0], translation_data.en_dict)
        predicted_token = predicted_str.split()[i]
        if predicted_token == '<PAD>':
            break
        joiner_char = ' ' if len(output) > 0 else ''
        output = output + joiner_char + predicted_str.split()[i]
    
    print(f'DE: {input_string}\nEN: {output}')

##################################################################################################################################
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_bool("plot", False, "Plot the model based on model codes")
flags.DEFINE_bool("smoke", False, "Try train a little bit to do smoking test")
flags.DEFINE_bool("train", False, "Train the model and save")
flags.DEFINE_bool("predict", False, "Load saved model and predict")
flags.DEFINE_string("input", "es ist gut", "Used with --predict, German input, to be translated to English")
flags.DEFINE_string("tb_dir", "logs", "TensorbBoard log folder")

def main(unused_args):
    """
    Samples:
      python transformer.py --plot
      python transformer.py --smoke --tb_dir "logs"
      python transformer.py --train --tb_dir "logs"
      python transformer.py --predict --input "es ist gut" 
    """
    if FLAGS.plot:
        model = get_model()
        plot_model(model)
    if FLAGS.smoke:
        train(steps_per_epoch=10, epochs=2, tensorboard=True, tb_dir=FLAGS.tb_dir)
    if FLAGS.train:
        train(steps_per_epoch=None, epochs=2, tensorboard=True, tb_dir=FLAGS.tb_dir)
    if FLAGS.predict:
        predict(FLAGS.input)
        
    print('Bye...')

if __name__ == '__main__':
    app.run(main)

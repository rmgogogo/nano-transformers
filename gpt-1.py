import tensorflow as tf
import numpy as np
import codecs

class SimpleBookData(object):
    UNK = '<unk>'
    PAD = '<pad>'

    """
    SimpleBook data.
    Data is from https://arxiv.org/abs/1911.12391
    https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
    We also can reuse the Translation dataset used for Transformer. However that dataset is not easy to train the generative model.
    """
    def __init__(self, folder="data/simplebooks/simplebooks-92"):
        self.vocab, self.vocab_index = self._get_vocab(file=os.path.join(folder, 'train.vocab'))
        self.lines = self._get_lines(file=os.path.join(folder, 'train.txt'))
        
    def _get_vocab(self, file):
        """
        Load the dictionary from train.vocab. It contains '<unk>' and '<eob>'. Additionally, we put '<pad>' to the first place.
        """
        text = codecs.open(file, 'r', 'utf-8').read()
        lines = text.split('\n')
        vocab = [SimpleBookData.PAD]
        for line in lines:
            wordAndCount = line.split()
            if len(wordAndCount) > 0:
                vocab.append(wordAndCount[0])

        vocab_index = {}
        for idx in range(len(vocab)):
            word = vocab[idx]
            vocab_index[word] = idx

        return vocab, vocab_index

    def _get_lines(self, file):
        text = codecs.open(file, 'r', 'utf-8').read()
        return text.split('\n')
    
    def _generate_tokens(self, top_k_vocab):
        for line in self.lines:
            yield self.tokenize(sentence=line, top_k_vocab=top_k_vocab)
    
    def tokenize(self, sentence, top_k_vocab):
        words = sentence.split()
        tokens = []
        for word in words:
            idx = self.vocab_index[word] if word in self.vocab_index else self.vocab_index[SimpleBookData.UNK]
            if idx >= top_k_vocab:
                idx = self.vocab_index[SimpleBookData.UNK]
            tokens.append(idx)
        return tokens

    def detokenize(self, tokens):
        if len(tokens.shape) > 1:
            # not argmax yet, it's logits
            print('Not argmx yet, please do not input logits.')
            tokens = np.argmax(tokens, axis=-1)
        return " ".join([self.vocab[token] for token in tokens])

    def generate_pair(self, max_len, top_k_vocab, min_tokens_per_sample):
        """
        Generate x,y for Auto Reguression
        """
        for tokens in self._generate_tokens(top_k_vocab=top_k_vocab):
            start_idx = 0
            while start_idx < len(tokens):
                sub_tokens = tokens[start_idx:]
                # ignore the too short sub_tokens
                if len(sub_tokens) < min_tokens_per_sample:
                    break
                # padding sub_tokens to max_len+1
                if len(sub_tokens) < max_len+1:
                    sub_tokens += [self.vocab_index[SimpleBookData.PAD]]*(max_len+1-len(sub_tokens))

                yield sub_tokens[:max_len], sub_tokens[1:1+max_len]
                start_idx += max_len
    
    def get_generator(self, max_len, top_k_vocab, min_tokens_per_sample):
        def generator():
            return self.generate_pair(max_len=max_len, top_k_vocab=top_k_vocab, min_tokens_per_sample=min_tokens_per_sample)
        return generator
    
    def get_pad_index(self):
        return self.vocab_index[SimpleBookData.PAD]
    
    @staticmethod
    def smoke(max_len, top_k_vocab, min_tokens_per_sample):
        data = SimpleBookData()
        dataset = tf.data.Dataset.from_generator(
            data.get_generator(max_len=max_len,top_k_vocab=top_k_vocab, min_tokens_per_sample=min_tokens_per_sample),
            output_signature=((tf.TensorSpec(shape=(max_len), dtype=tf.int64), tf.TensorSpec(shape=(max_len), dtype=tf.int64))))
        for x in dataset:
            print(data.detokenize(x[0].numpy()))

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

def get_model(max_len, hidden, head, vocab_size):
    """
    Get the model
    """
    # Encode (max_len)
    # Input based on sparse index and then an embedding layer, it's much faster than one-hot.
    input = tf.keras.Input(shape=(max_len), name='input')
    data = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=hidden, name='input_embedding')(input)
    data = RmPosition(max_len, hidden, name='input_positioning')(data)

    for i in range(2):
        #===== Transformer Decode Block Starts =====
        # Decode Attention Add Norm
        weighted_v = RmMultiHeadAttention(head=head, hidden=hidden, sequence_mask=True, name=f'decode_attention-{i}')((data, data, data))
        data = tf.keras.layers.Add(name=f'decode_attention_add-{i}')([data, weighted_v])
        data =  tf.keras.layers.LayerNormalization(axis=-1, name=f'decode_attention_norm-{i}')(data)
        
        # Decode FeedForward Add Norm
        decodeIncreased = tf.keras.layers.Dense(units=hidden*2, activation='relu', name=f'decode_ff_increse-{i}')(data)
        decodeDecreased = tf.keras.layers.Dense(units=hidden, activation=None, name=f'decode_ff_decrease-{i}')(decodeIncreased)
        data = tf.keras.layers.Add(name=f'decode_ff_add-{i}')([data, decodeDecreased])
        data = tf.keras.layers.LayerNormalization(axis=-1, name=f'decode_ff_norm-{i}')(data)
        #===== Transformer Decode Block Ends =====

    # Output (logits, not softmax, loss-fn side will take care it.)
    output = tf.keras.layers.Dense(vocab_size, activation=None, name='output')(data)

    model = tf.keras.Model(inputs=input, outputs=output, name='model')
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

import os
import datetime

def smoke_data(max_len, top_k_vocab, min_tokens_per_sample):
    SimpleBookData.smoke(max_len=max_len, top_k_vocab=top_k_vocab, min_tokens_per_sample=min_tokens_per_sample)

def train(max_len, top_k_vocab, min_tokens_per_sample, hidden, head=4, batch_size=64, epochs=1, steps_per_epoch=None, tensorboard=False, tb_dir='logs', model_dir='saved_model/gpt_1_pretrain'):
    """
    Train the model
    """
    data = SimpleBookData()
    dataset = tf.data.Dataset.from_generator(
        data.get_generator(max_len=max_len, top_k_vocab=top_k_vocab, min_tokens_per_sample=min_tokens_per_sample),
        output_signature=((tf.TensorSpec(shape=(max_len), dtype=tf.int64), tf.TensorSpec(shape=(max_len), dtype=tf.int64))))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = get_model(max_len=max_len, hidden=hidden, head=head, vocab_size=top_k_vocab)
    if tensorboard:
        tb_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=4*1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    print(f'Launch TensorBoard to check the logs:\n  tensorboard --logdir {tb_dir}')

    callbacks = []
    if tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
    _ = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks) 
    model.save(model_dir)

def predict(input_string, max_len, top_k_vocab, max_predict=80, model_dir='saved_model/gpt_1_pretrain'):
    model=tf.keras.models.load_model(model_dir)
    data = SimpleBookData()

    output_str = ''
    current_tokens = data.tokenize(sentence=input_string, top_k_vocab=top_k_vocab)
    num_predicted = 0
    while num_predicted < max_predict:
        x = current_tokens[-max_len:]
        lastOutputTokenIdx = min(len(current_tokens), max_len) - 1
        if len(x) < max_len:
            x = x + [0] * (max_len - len(x))
        predicted = model.predict(np.array([x]), verbose = 0)
        logits = predicted[0][lastOutputTokenIdx]
        predicted_token = np.argmax(logits, axis=-1)
        current_tokens.append(predicted_token)
        num_predicted += 1
        
        predicted_word = data.vocab[predicted_token]
        if predicted_word == SimpleBookData.PAD:
            break
        output_str = output_str + ' ' + predicted_word    
    print(f'{input_string}{output_str}')

##################################################################################################################################
from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_bool("plot", False, "Plot the model based on model codes")
flags.DEFINE_bool("sample", False, "Sample a few data for checking")
flags.DEFINE_bool("smoke", False, "Try train a little bit to do smoking test")
flags.DEFINE_bool("train", False, "Train the model and save")
flags.DEFINE_bool("predict", False, "Load saved model and predict")
flags.DEFINE_string("input", "I want to", "Used with --predict, English input, it's prompt, predict next tokens")
flags.DEFINE_string("tb_dir", "/tmp/logs", "TensorbBoard log folder")
flags.DEFINE_integer("max_len", 30, "Max senquence length, the max number of tokens")
flags.DEFINE_integer("epochs", 1, "Epochs to train")
flags.DEFINE_integer("vocab", 10000, "Vocab size, choose top-k vocab words")
flags.DEFINE_integer("hidden", 512, "hidden vector size")
flags.DEFINE_integer("min_tokens_per_sample", 8, "ignore sequence which is shorter than it")

def main(unused_args):
    """
    Samples:
      python gpt-1.py --sample
      python gpt-1.py --plot
      python gpt-1.py --smoke
      python gpt-1.py --train
      python gpt-1.py --train --vocab 10000 --max_len 30 --hidden 512 --epochs 1
      python gpt-1.py --predict --input "I want to" 
    """

    import random
    import time
    random.seed(time.time())

    if FLAGS.plot:
        model = get_model(max_len=FLAGS.max_len, vocab_size=FLAGS.vocab, hidden=FLAGS.hidden, head=4)
        print(model.summary())
        plot_model(model)
    if FLAGS.sample:
        smoke_data(max_len=FLAGS.max_len, top_k_vocab=FLAGS.vocab, min_tokens_per_sample=FLAGS.min_tokens_per_sample)
    if FLAGS.smoke:
        train(max_len=FLAGS.max_len, top_k_vocab=FLAGS.vocab, min_tokens_per_sample=FLAGS.min_tokens_per_sample, hidden=FLAGS.hidden, steps_per_epoch=100, epochs=FLAGS.epochs, tensorboard=False, tb_dir=FLAGS.tb_dir)
    if FLAGS.train:
        train(max_len=FLAGS.max_len, top_k_vocab=FLAGS.vocab, min_tokens_per_sample=FLAGS.min_tokens_per_sample, hidden=FLAGS.hidden, steps_per_epoch=None, epochs=FLAGS.epochs, tensorboard=True, tb_dir=FLAGS.tb_dir)
    if FLAGS.predict:
        predict(input_string=FLAGS.input, max_len=FLAGS.max_len, top_k_vocab=FLAGS.vocab)

if __name__ == '__main__':
    app.run(main)

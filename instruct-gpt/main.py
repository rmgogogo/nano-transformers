from absl import flags, app
import os
import datetime
import tensorflow as tf
import numpy as np

from rm_model import get_model, plot_model
from rm_data import MathData
from rm_tokenizer import RmTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_bool("sample", False, "Sample a few data for checking")
flags.DEFINE_bool("plot", False, "Plot the model based on model codes")
flags.DEFINE_bool("smoke", False, "Try train a little bit to do smoking test")
flags.DEFINE_bool("train", False, "Train the model and save")
flags.DEFINE_bool("predict", False, "Load saved model and predict")
flags.DEFINE_string("input", "1 + 1 =", "Used with --predict, English input, it's prompt, predict next tokens")
flags.DEFINE_string("tb_dir", "/tmp/logs", "TensorbBoard log folder")
flags.DEFINE_integer("max_len", 5, "Max senquence length, the max number of tokens")
flags.DEFINE_integer("hidden", 64, "hidden vector size")
flags.DEFINE_integer("head", 8, "attention heads")
flags.DEFINE_integer("n_blocks", 8, "how many Transformer Decoder blocks")
flags.DEFINE_integer("epochs", 20, "Epochs to train")
flags.DEFINE_integer("smoke_steps", 2000, "how many steps to do smoking training")
flags.DEFINE_integer("add_normal_repeat", 100, "repeat the normal-add samples")
flags.DEFINE_float("dropout", 0.1, "drop out rate, 0-1")

def train(n_blocks, max_len, hidden, head, dropout_rate, add_normal_repeat, batch_size=64, epochs=1, steps_per_epoch=None, tensorboard=False, tb_dir='/tmp/instruct-gpt/tb', model_dir='model.keras'):
    """
    Train the model
    """
    data = MathData()
    tokenizer = data.get_tokenizer()
    dataset = tf.data.Dataset.from_generator(
        data.get_generator_normal(max_len=max_len),
        output_signature=((tf.TensorSpec(shape=(max_len), dtype=tf.int64), tf.TensorSpec(shape=(max_len), dtype=tf.int64))))
    dataset = dataset.repeat(add_normal_repeat).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    model = get_model(n_blocks=n_blocks, max_len=max_len, hidden=hidden, head=head, vocab_size=tokenizer.get_vocab_size(), dropout_rate=dropout_rate)
    if tensorboard:
        tb_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f'Launch TensorBoard to check the logs:\n  tensorboard --logdir {tb_dir}')

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = []
    if tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
    _ = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks) 
    model.save(model_dir)


def predict(input_string, model_dir='model.keras'):
    tokenizer = MathData().get_tokenizer()

    model=tf.keras.models.load_model(model_dir)
    max_len = model.get_layer('input').output.shape[1]

    output_str = ''
    current_tokens = tokenizer.encode_sentence(input_string)
    num_predicted = 0
    while num_predicted < 16:
        x = current_tokens[-max_len:]
        lastOutputTokenIdx = min(len(current_tokens), max_len) - 1
        if len(x) < max_len:
            x = x + [0] * (max_len - len(x))
        
        predicted = model.predict(np.array([x]), verbose = 0)
        logits = predicted[0][lastOutputTokenIdx]
        predicted_token = np.argmax(logits, axis=-1)
        current_tokens.append(predicted_token)
        num_predicted += 1
        
        predicted_word = tokenizer.decode(predicted_token)
        if predicted_word == RmTokenizer.PAD:
            break
        output_str = output_str + ' ' + predicted_word    
        print('prob: ', max(np.exp(logits)/np.exp(logits).sum()))
    print(f'{input_string}{output_str}')

def main(unused_args):
    """
    Samples:
      python main.py --sample --plot --smoke --predict
      python main.py --train --predict
      python main.py --predict --input "9 + 9 ="
    """
    if FLAGS.sample:
        MathData.smoke(max_len=FLAGS.max_len)
    if FLAGS.plot:
        data = MathData()
        tokenizer = data.get_tokenizer()
        model = get_model(n_blocks=FLAGS.n_blocks, max_len=FLAGS.max_len, hidden=FLAGS.hidden, head=FLAGS.head, vocab_size=tokenizer.get_vocab_size(), dropout_rate=FLAGS.dropout)
        print(model.summary())
        plot_model(model)
    if FLAGS.smoke:
        train(n_blocks=FLAGS.n_blocks, max_len=FLAGS.max_len, hidden=FLAGS.hidden, head=FLAGS.hidden,
              add_normal_repeat=FLAGS.add_normal_repeat,
              dropout_rate=FLAGS.dropout, steps_per_epoch=FLAGS.smoke_steps, epochs=FLAGS.epochs, tensorboard=False, tb_dir=FLAGS.tb_dir)
    if FLAGS.train:
        train(n_blocks=FLAGS.n_blocks, max_len=FLAGS.max_len, hidden=FLAGS.hidden, head=FLAGS.hidden,
              add_normal_repeat=FLAGS.add_normal_repeat,
              dropout_rate=FLAGS.dropout, steps_per_epoch=None, epochs=FLAGS.epochs, tensorboard=True, tb_dir=FLAGS.tb_dir)
    if FLAGS.predict:
        predict(input_string=FLAGS.input)

if __name__ == '__main__':
    app.run(main)

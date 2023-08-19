import tensorflow as tf
from rm_tokenizer import RmTokenizer

class MathData:
    """
    Generates a set of data for learning math and do reasoning.
    
    Samples:
    1 + 1 = 2
    1 + 2 = 3
    0 + 1 = 1
    """
    def __init__(self):
        self.tokenizer = None

    def generate_add_normal(self):
        """
        1 = 1
        1 + 2 = 3
        2 + 1 = 3
        3 = 1 + 2
        3 = 2 + 1
        """
        for a in range(1, 10):
            yield f'{a} = {a}'
            for b in range(1, 10):
                yield f'{a} + {b} = {a+b}'
                yield f'{b} + {a} = {a+b}'
                yield f'{a+b} = {a} + {b}'
                yield f'{a+b} = {b} + {a}'

    def generate_add_zero(self):
        """
        0 + 1  = 2
        """
        for a in range(1, 5):
            yield f'0 + {a} = {a}'

    def get_tokenizer(self):
        """
        Get tokenizer which contains dictionary.
        """
        if self.tokenizer == None:
            tokenizer = RmTokenizer()
            for sentence in self.generate_add_normal():
                tokenizer.visit_words(sentence.split())
            for sentence in self.generate_add_zero():
                tokenizer.visit_words(sentence.split())
            self.tokenizer = tokenizer
        return self.tokenizer

    def generate_pair_normal(self, max_len):
        """
        Generate x,y for Auto Reguression
        """
        tokenizer = self.get_tokenizer()
        padding_token = tokenizer.encode(RmTokenizer.PAD)
        for sentence in self.generate_add_normal():
            tokens = [tokenizer.encode(word) for word in sentence.split()]
            padding_len = max_len + 1 - len(tokens)
            tokens = tokens + [padding_token]*padding_len
            yield tokens[:max_len], tokens[1:1+max_len]

    def get_generator_normal(self, max_len):
        def generator():
            return self.generate_pair_normal(max_len=max_len)
        return generator
    
    @staticmethod
    def smoke(max_len=10):
        data = MathData()
        tokenizer = data.get_tokenizer()
        dataset = tf.data.Dataset.from_generator(
            data.get_generator_normal(max_len=max_len),
            output_signature=((tf.TensorSpec(shape=(max_len), dtype=tf.int64), tf.TensorSpec(shape=(max_len), dtype=tf.int64))))
        for xy in dataset:
            print(tokenizer.decode_tokens(xy[0].numpy()), "$$$$$", tokenizer.decode_tokens(xy[1].numpy()))


if __name__ == '__main__':
    data = MathData()
    tokenizer = data.get_tokenizer()
    for sentence in data.generate_add_normal():
        tokenizer.visit_words(sentence.split())
    for sentence in data.generate_add_zero():
        tokenizer.visit_words(sentence.split())
    
    sentence = next(data.generate_add_normal())
    tokens = [tokenizer.encode(word) for word in sentence.split()]
    print(sentence)
    print(tokens)

    sentence = next(data.generate_add_zero())
    tokens = [tokenizer.encode(word) for word in sentence.split()]
    print(sentence)
    print(tokens)

    MathData.smoke(max_len=8)

class RmTokenizer():
    PAD = '<p>'
    UNK = '<u>'

    def __init__(self):
        self.words = []
        self.word_index = {}
        self.visit(RmTokenizer.PAD)
        self.visit(RmTokenizer.UNK)

    def get_vocab_size(self):
        return len(self.words)

    def visit(self, word):
        if word not in self.word_index:
            self.words.append(word)
            self.word_index[word] = len(self.words)-1
    
    def visit_words(self, words):
        for word in words:
            self.visit(word)
    
    def encode(self, word):
        if word not in self.word_index:
            word = RmTokenizer.UNK
        return self.word_index[word]
    
    def decode(self, token):
        if token >= len(self.words):
            return RmTokenizer.UNK
        else:
            return self.words[token]

    def encode_sentence(self, sentence):
        return [self.encode(word) for word in sentence.split()]

    def decode_tokens(self, tokens):
        words = [self.decode(token) for token in tokens]
        return ' '.join(words)

if __name__ == '__main__':
    tokenizer = RmTokenizer()
    for i in range(10):
        tokenizer.visit(i)
    
    for i in range(10):
        word = tokenizer.decode(i)
        print(i, word, tokenizer.encode(word))
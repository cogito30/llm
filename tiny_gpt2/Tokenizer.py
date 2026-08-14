import re
import tiktoken

def create_vocab(file_path="the-verdict.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab, vocab_size

class SimpleTokenizer:
    def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [ item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"\'])', r'\1', text)
        return text

class Tokenizer:
    def __init__(self, model="gpt2"):
        self.tokenizer = tiktoken.get_encoding(model)

    def encode(self, text, allowed_specials={"<|endoftext|>"}):
        encoded = self.tokenizer.encode(text, allowed_special=allowed_specials)
        return encoded

    def decode(self, token_ids):
        decoded = self.tokenizer.decode(token_ids)
        return decoded

if __name__ == "__main__":
    vocab, vocab_size = create_vocab("the-verdict.txt")
    # print(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sulit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    # print(text)

    print("---Test SimpleToknenizer---")
    tokenizer = SimpleTokenizer(vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

    print("---Test Tokenizer---")
    text = "Hello, do you like tea? <|endoftext|> In the sulit terraces of someunkownPalace."
    tokenizer = Tokenizer("gpt2")
    encoded = tokenizer.encode(text, allowed_specials={"<|endoftext|>"})
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)


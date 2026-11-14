from src.tokenizer.BPE import BPE
from src.data.load_data import load_data


class Tokenizer:
    def __init__(self, model, dataset):
        self.tokenizer = model
        self.dataset = dataset

    def tokenize(self):
        for i in range(0, 10):
            print(self.dataset["train"][i])

if __name__ == '__main__':
    dataset = load_data()
    tokenizer = Tokenizer(model=BPE(), dataset=dataset)
    tokenizer.tokenize()
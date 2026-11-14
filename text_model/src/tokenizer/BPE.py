from src.data.load_data import load_data

class BPE:
    def __init__(self):
        self.dataset = load_data()

    def test_dataset(self):
        print(self.dataset['train'][0])

if __name__ == "__main__":
    _BPE = BPE()
    _BPE.test_dataset()
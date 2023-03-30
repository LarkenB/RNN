import string
from rnn import CharRNN


N_EPOCHS = 64
CHUNK_LEN = 200
BATCH_SIZE = 100
FILENAME = "data/tiny-shakespeare.txt"
LEARN_RATE = 0.01
HIDDEN_SIZE = 100
N_LAYERS = 2
ALL_CHARS = string.printable
N_CHARS = len(ALL_CHARS)


def read_file(filename):
    file = open(filename, 'r', encoding='utf-8')
    data = file.read()
    file.close()
    return data


def main():
    data = read_file(FILENAME)
    vocab = set(data)

    model = CharRNN(
        vocab,
        HIDDEN_SIZE,
        N_LAYERS,
    )

    model.learn(data, N_EPOCHS, LEARN_RATE)
    print(model.generate(start="QUEEN:"))


if __name__ == '__main__':
    main()

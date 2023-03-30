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
    model = CharRNN(
        N_CHARS,
        HIDDEN_SIZE,
        N_CHARS,
        n_layers=N_LAYERS,
    )

    data = read_file(FILENAME)
    model.learn(data, N_EPOCHS, LEARN_RATE)
    print(model.generate(start="QUEEN:"))


if __name__ == '__main__':
    main()

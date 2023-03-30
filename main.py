from rnn import CharRNN

HIDDEN_SIZE = 100
N_LAYERS = 2
LEARN_RATE = 0.01
N_EPOCHS = 60


def read_shakespeare(path="data/tiny-shakespeare.txt"):
    file = open(path, 'r', encoding='utf-8')
    data = file.read()
    file.close()
    return data


def main():
    data = read_shakespeare()
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

import string
from random import random
from rnn import CharRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import unidecode


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
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = ALL_CHARS.index(string[c])
        except:
            continue
    return tensor


def random_training_set(file, file_len, chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)

    return inp, target


def train(decoder, criterion, decoder_optimizer, inp, target):
    hidden = decoder.init_hidden(BATCH_SIZE)

    decoder.zero_grad()
    loss = 0

    for c in range(CHUNK_LEN):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(BATCH_SIZE, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data / CHUNK_LEN


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = ALL_CHARS[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))

    return predicted


def main():

    decoder = CharRNN(
        N_CHARS,
        HIDDEN_SIZE,
        N_CHARS,
        n_layers=N_LAYERS,
    )
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARN_RATE)
    criterion = nn.CrossEntropyLoss()

    loss_avg = 0

    file, file_len = read_file(FILENAME)

    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(N_EPOCHS):
        inp, target = random_training_set(file, file_len, CHUNK_LEN, BATCH_SIZE)
        loss = train(decoder, criterion, decoder_optimizer, inp, target)
        print(f'Epoch: {epoch} | Loss: {loss}')
        loss_avg += loss

    print(generate(decoder, prime_str="QUEEN:"))


if __name__ == '__main__':
    main()

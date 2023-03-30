import random
import string
import torch
import torch.nn as nn

SEQ_SIZE = 200
BATCH_SIZE = 100


# TODO: Fix loss and how its handled, add comments, refactor var names,
#       change whatever else as needed, remove string dependency


class CharRNN(nn.Module):
    def __init__(self, vocab, hidden_size, n_layers):
        super(CharRNN, self).__init__()
        self.vocab = list(vocab)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(len(vocab), hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, len(vocab))

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.fc(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

    def char_tensor(self, text):
        return torch.tensor([self.vocab.index(c) for c in text])

    def random_training_set(self, data, chunk_len, batch_size):
        data_len = len(data)
        inp = torch.LongTensor(batch_size, chunk_len)
        target = torch.LongTensor(batch_size, chunk_len)
        for bi in range(batch_size):
            start_index = random.randint(0, data_len - chunk_len)
            end_index = start_index + chunk_len + 1
            chunk = data[start_index:end_index]
            inp[bi] = self.char_tensor(chunk[:-1])
            target[bi] = self.char_tensor(chunk[1:])

        return inp, target

    def learn(self, data, n_epochs, learn_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            inp, target = self.random_training_set(data, SEQ_SIZE, BATCH_SIZE)
            loss = self.epoch(criterion, optimizer, inp, target)
            print(f'Epoch: {epoch}/{n_epochs} | Loss: {loss}')
        return

    def epoch(self, criterion, optimizer, inp, target):
        hidden = self.init_hidden(BATCH_SIZE)

        self.zero_grad()
        loss = 0

        for c in range(SEQ_SIZE):
            output, hidden = self(inp[:, c], hidden)
            loss += criterion(output.view(BATCH_SIZE, -1), target[:, c])

        loss.backward()
        optimizer.step()

        return loss.data / SEQ_SIZE

    def generate(self, start='QUEEN:', predict_len=100, temperature=0.8):
        hidden = self.init_hidden(1)
        prime_input = self.char_tensor(start).unsqueeze(0)
        predicted = start

        # Use priming string to "build up" hidden state
        for p in range(len(start) - 1):
            _, hidden = self(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = self.vocab[top_i]
            predicted += predicted_char
            inp = self.char_tensor(predicted_char).unsqueeze(0)

        return predicted

import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

        # Hidden state
        self.h = np.zeros((hidden_size, 1))

    def forward(self, x):
        self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
        y = np.dot(self.Why, self.h) + self.by
        return y

    def reset_hidden_state(self):
        self.h = np.zeros_like(self.h)

def preprocess_text(text):
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char, chars

def text_to_one_hot(text, char_to_idx):
    # Convert text to one-hot encoding
    one_hot = np.zeros((len(char_to_idx), 1))
    if text in char_to_idx:
        one_hot[char_to_idx[text]] = 1
    return one_hot

def generate_text(rnn, start_char, char_to_idx, idx_to_char, length=100):
    rnn.reset_hidden_state()
    x = text_to_one_hot(start_char, char_to_idx)
    generated_text = start_char

    for _ in range(length):
        y = rnn.forward(x)
        predicted_idx = np.argmax(y)
        next_char = idx_to_char[predicted_idx]
        generated_text += next_char
        x = text_to_one_hot(next_char, char_to_idx)

    return generated_text

def train_rnn(rnn, training_text, char_to_idx, idx_to_char, epochs):
    for epoch in range(epochs):
        # (Training logic goes here)
        # For now, this is just a placeholder to demonstrate the concept.
        
        # Print a message every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} completed.")

    print("Training complete.")

def main():
    # Read training text from an external file
    with open("dictionary.txt", "r") as f:
        training_text = f.read()

    char_to_idx, idx_to_char, chars = preprocess_text(training_text)
    input_size = len(chars)
    hidden_size = 50  # Size of hidden layer
    output_size = len(chars)

    # Initialize RNN
    rnn = SimpleRNN(input_size, hidden_size, output_size)

    # Get number of epochs from user
    epochs = int(input("Enter the number of epochs for training: "))
    
    # Train the RNN
    train_rnn(rnn, training_text, char_to_idx, idx_to_char, epochs)

    # Interactive mode
    print("Chat with the RNN! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Generate response based on input
        response = generate_text(rnn, user_input[0], char_to_idx, idx_to_char, length=50)
        print("RNN: " + response)

if __name__ == "__main__":
    main()  # Run the main function

# 123

# chat_learning.py
import numpy as np

def update_training_data(existing_data, new_input):
    """ Append new input to existing training data and return. """
    return existing_data + new_input + '\n'

def save_training_data(file_path, data):
    """ Save the updated training data to a file. """
    with open(file_path, 'w') as f:
        f.write(data)

def load_training_data(file_path):
    """ Load training data from a file. """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

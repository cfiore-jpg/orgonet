import numpy as np
import tensorflow as tf
from model import OrgoNet
import pickle
from model import loss_function, accuracy_function
from position_generator import PositionGenerator 
import re
import pickle


if __name__ == '__main__':

    names_file = 'data/train/train_names.pkl'
    elements_file = 'data/train/train_elements.pkl'
    positions_file = 'data/train/train_positions_tensor.pkl'

    # names_file = 'data/test/test_names.pkl'
    # elements_file = 'data/test/test_elements.pkl'
    # positions_file = 'data/test/test_positions_tensor.pkl'
    
    names_vocab_file = 'data/names_vocabulary.pkl'
    elements_vocab_file = 'data/elements_vocabulary.pkl'

    weights_path = 'data/saved_weights5/orgonet'

    with open(names_file, 'rb') as f:
        names = tf.convert_to_tensor(pickle.load(f), dtype=tf.float32)

    with open(elements_file, 'rb') as f:
        elements = tf.convert_to_tensor(pickle.load(f), dtype=tf.float32)

    with open(positions_file, 'rb') as f:
        positions = pickle.load(f)

    with open(names_vocab_file, 'rb') as f:
        names_vocab = pickle.load(f)

    with open(elements_vocab_file, 'rb') as f:
        elements_vocab = pickle.load(f)  


    ## Model Parameters
    hidden_size = 128
    optimizer = tf.optimizers.Adam(0.001)
    loss = loss_function
    metrics = [accuracy_function]

    name_vocab_size = len(names_vocab)
    element_vocab_size = len(elements_vocab)
    max_name_size = names.shape[1]
    max_molecule_size = elements.shape[1]

    ## Model setup
    orgonet = OrgoNet(name_vocab_size, element_vocab_size, hidden_size, max_name_size, max_molecule_size)
    orgonet.compile(optimizer, loss, metrics)

    ## Train Parameters
    pad_index = elements_vocab['<pad>'] 
    num_epochs = 20
    batch_size = 1024

    ## Train and save
    orgonet.train(names, elements, positions, pad_index, num_epochs, batch_size)
    orgonet.save_weights(weights_path)

    ## Load and test
    # orgonet.load_weights(weights_path)
    # orgonet.test(names, elements, positions, pad_index, 1024)


    # orgonet.position_finder(tf.expand_dims(names[0], 0), tf.expand_dims(elements[0], 0))
    # orgonet.position_finder.summary()
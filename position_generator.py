import tensorflow as tf

try: from transformer import TransformerEncoder, TransformerDecoder, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

class PositionGenerator(tf.keras.Model):

    def __init__(self, name_vocab_size, element_vocab_size, hidden_size, max_name_size, max_molecule_size):

        super().__init__()
        self.name_vocab_size  = name_vocab_size
        self.element_vocab_size = element_vocab_size
        self.hidden_size = hidden_size
        self.max_name_size = max_name_size
        self.max_molecule_size = max_molecule_size

        self.name_encoding = PositionalEncoding(self.name_vocab_size, self.hidden_size, self.max_name_size)
        self.encoder_layer = TransformerEncoder(hidden_size)

        self.element_encoding = PositionalEncoding(self.element_vocab_size, self.hidden_size, self.max_molecule_size)
        self.decoder_layer = TransformerDecoder(hidden_size)

        self.position_finder = tf.keras.layers.Dense(3)

    def call(self, names, elements):
        encoded_names = self.name_encoding(names)
        context = self.encoder_layer(encoded_names)

        encoded_elements = self.element_encoding(elements)
        decoded_elements = self.decoder_layer(encoded_elements, context)

        positions = self.position_finder(decoded_elements)
        return positions

    def get_config(self):
        return {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "window_size": self.window_size}
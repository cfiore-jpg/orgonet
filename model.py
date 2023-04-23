import numpy as np
import tensorflow as tf

class OrgoNet(tf.keras.Model):

    def __init__(self, position_finder, **kwargs):
        super().__init__(**kwargs)
        self.position_finder = position_finder

    @tf.function
    def call(self, names, elements):
        return self.position_finder(names, elements)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_names, train_elements, train_locations, pad_index, batch_size=30):

        num_batches = int(len(train_names) / batch_size)

        indices = tf.random.shuffle(np.arange(0, len(train_names), 1))
        train_names = tf.gather(train_names, indices)
        train_elements = tf.gather(train_elements, indices)
        train_locations = tf.gather(train_locations, indices)

        total_loss, total_seen, total_correct = 0, 0, 0
        for index, end in enumerate(range(batch_size, len(train_names)+1, batch_size)):

            start = end - batch_size
            batch_names = train_names[start:end, :]
            batch_elements = train_elements[start:end, :]
            batch_locations = train_locations[start:end, :, :]

            with tf.GradientTape() as tape:
                pred_locations = self(batch_names, batch_elements)
                mask = batch_names != pad_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(pred_locations, batch_locations, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            accuracy = self.accuracy_function(pred_locations, batch_locations, mask)

            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Train {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')
    
        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        # num_batches = int(len(test_captions) / batch_size)

        # total_loss, total_seen, total_correct = 0, 0, 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

        #     # NOTE: 
        #     # - The captions passed to the decoder should have the last token in the window removed:
        #     #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
        #     #
        #     # - When computing loss, the decoder labels should have the first word removed:
        #     #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

        #     ## Get the current batch of data, making sure to try to predict the next word
        #     start = end - batch_size
        #     batch_image_features = test_image_features[start:end, :]
        #     decoder_input = test_captions[start:end, :-1]
        #     decoder_labels = test_captions[start:end, 1:]

        #     ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
        #     probs = self(batch_image_features, decoder_input)
        #     mask = decoder_labels != padding_index
        #     num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
        #     loss = self.loss_function(probs, decoder_labels, mask)
        #     accuracy = self.accuracy_function(probs, decoder_labels, mask)

        #     ## Compute and report on aggregated statistics
        #     total_loss += loss
        #     total_seen += num_predictions
        #     total_correct += num_predictions * accuracy

        #     avg_loss = float(total_loss / total_seen)
        #     avg_acc = float(total_correct / total_seen)
        #     avg_prp = np.exp(avg_loss)
        #     print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc
    
    def get_config(self):
        return {"decoder": self.decoder}


def accuracy_function(predictions, locations, mask):
    masked_predictions = tf.boolean_mask(predictions, mask)
    masked_locations = tf.boolean_mask(locations, mask)
    acc = tf.keras.metrics.mean_squared_error(masked_predictions, masked_locations)
    return acc


def loss_function(predictions, locations, mask):
    masked_predictions = tf.boolean_mask(predictions, mask)
    masked_locations = tf.boolean_mask(locations, mask)
    loss = tf.keras.losses.mean_squared_error(masked_predictions, masked_locations)
    return loss
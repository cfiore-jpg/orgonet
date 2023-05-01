import numpy as np
from position_generator import PositionGenerator
import tensorflow as tf

class OrgoNet(tf.keras.Model):

    def __init__(self, name_vocab_size, element_vocab_size, hidden_size, max_name_size, max_molecule_size):
        super().__init__()
        self.position_finder = PositionGenerator(name_vocab_size, element_vocab_size, hidden_size, max_name_size, max_molecule_size)


    def call(self, names, elements):
        return self.position_finder(names, elements)  


    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]


    def train(self, train_names, train_elements, train_locations, pad_index, num_epochs, batch_size):

        num_batches = int(len(train_names) / batch_size)

        final_loss = 0
        for epoch in range(num_epochs):

            indices = tf.random.shuffle(np.arange(0, len(train_names), 1))
            train_names = tf.gather(train_names, indices)
            train_elements = tf.gather(train_elements, indices)
            train_locations = tf.gather(train_locations, indices)

            total_loss = 0
            for index, end in enumerate(range(batch_size, len(train_names)+1, batch_size)):

                start = end - batch_size
                batch_names = train_names[start:end, :]
                batch_elements = train_elements[start:end, :]
                batch_locations = train_locations[start:end, :, :]

                mask = batch_elements != pad_index
                with tf.GradientTape() as tape:
                    pred_locations = self.call(batch_names, batch_elements)
                    loss = self.loss_function(pred_locations, batch_locations, mask)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                total_loss += loss

                print(f"[Epoch: {epoch+1},   Batch: {index+1}/{num_batches}] --> Loss: {loss:.3f}", end='\r')


            final_loss = total_loss / num_batches

            print(f"\n[Epoch: {epoch+1}] --> Average Loss: {final_loss:.3f}\n")
    
        return final_loss


    def test(self, test_names, test_elements, test_locations, pad_index, batch_size):
        num_batches = int(len(test_names) / batch_size)

        total_loss = 0
        for index, end in enumerate(range(batch_size, len(test_names)+1, batch_size)):

            start = end - batch_size
            batch_names = test_names[start:end, :]
            batch_elements = test_elements[start:end, :]
            batch_locations = test_locations[start:end, :, :]

            mask = batch_elements != pad_index
            pred_locations = self.call(batch_names, batch_elements)
            loss = self.accuracy_function(pred_locations, batch_locations, mask)
            
            total_loss += loss

            print(f"[Batch: {index+1}/{num_batches}] --> Loss: {loss:.3f}", end='\r')

        final_loss = total_loss / num_batches

        print(f"\nAverage Testing Loss: {final_loss:.3f}\n")
    
        return final_loss

    
    def get_config(self):
        return {"decoder": self.decoder}


def accuracy_function(predictions, locations, mask):
    masked_predictions = tf.boolean_mask(predictions, mask)
    masked_locations = tf.boolean_mask(locations, mask)
    acc = tf.keras.metrics.mean_squared_error(masked_predictions, masked_locations)
    acc = tf.reduce_mean(acc)
    return acc


def loss_function(predictions, locations, mask):
    masked_predictions = tf.boolean_mask(predictions, mask)
    masked_locations = tf.boolean_mask(locations, mask)
    mse = tf.keras.losses.mean_squared_error(masked_predictions, masked_locations)
    loss = tf.reduce_mean(mse)
    return loss
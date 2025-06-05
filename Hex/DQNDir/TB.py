import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, log_dir='logs', **kwargs):
        # Initialize the parent TensorBoard class with the provided log_dir and kwargs
        super().__init__(log_dir=log_dir, **kwargs)
        self.step = 1  # Custom step counter for manual control
        # Set _train_dir explicitly to the 'train' subdirectory of log_dir
        self._train_dir = os.path.join(self.log_dir, 'train')
        self._should_write_train_graph = False
        # Add any other custom initialization here

    def set_model(self, model):
        # Override to prevent the parent class from setting up default writers
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Update stats with the logs from the epoch
        logs = logs or {}
        self.update_stats(**logs)
        self.step += 1  # Increment the step manually

    def on_batch_end(self, batch, logs=None):
        # No action needed at batch end
        pass

    def on_train_end(self, _):
        # No action needed at training end
        pass

    def update_stats(self, **stats):
        # Use the parent class's _train_writer to log metrics
        with self._train_writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=self.step)
            self._train_writer.flush()

# Example usage in your DQN training code:
# agent.model.fit(..., callbacks=[ModifiedTensorBoard(log_dir='path/to/logs')])
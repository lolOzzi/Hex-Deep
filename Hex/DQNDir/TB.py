import os
import tensorflow as tf

class ModifiedTensorBoard:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.step = 1
        self.writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, step=self.step)
            self.writer.flush()
import tensorflow as tf

print("TF version:            ", tf.__version__)
print("Built with CUDA?       ", tf.test.is_built_with_cuda())
print("GPU device name:       ", tf.test.gpu_device_name())        # should be "/device:GPU:0"
print("Num physical GPUs:     ", len(tf.config.list_physical_devices('GPU')))
print("Any GPU (tf.config)?   ", tf.config.list_logical_devices('GPU'))


# Place on GPU explicitly
with tf.device('/GPU:0'):
    # Create two large random tensors (e.g. 4096×4096) so GPU load is visible
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    c = tf.matmul(a, b)

# Run once to force kernel launch
_ = c.numpy()
print("Done one matmul on GPU")
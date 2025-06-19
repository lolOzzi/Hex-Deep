import numpy as np
import tensorflow as tf
from DQNNFD import NoisyFactorisedDense
import os
import shutil

# --- Configuration ---
KERAS_MODEL_PATH = 'models/5x5-simple_____4.90max____0.10avg____0.00min__1750211228.keras'
EXPORT_DIR = "test_simple_hex_dqn"

# --- Model Definitions (Architectures MUST be a 1-to-1 match) ---

def create_training_model():
    """ The original model with custom Noisy layers. """
    board_input = tf.keras.layers.Input(shape=(5, 5, 2), name='board_input')
    swap_input = tf.keras.layers.Input(shape=(1,), name='swap_input')
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='local_patterns')(board_input)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='global_patterns')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='pattern_combinations')(x)
    x_flat = tf.keras.layers.Flatten(name='flatten')(x)
    concatenated = tf.keras.layers.Concatenate(name='concatenate')([x_flat, swap_input])
    d = NoisyFactorisedDense(256, name='decision_layer_1')(concatenated)
    d = tf.keras.layers.ReLU(name='re_lu_1')(d)
    d = NoisyFactorisedDense(128, name='decision_layer_2')(d)
    d = tf.keras.layers.ReLU(name='re_lu_2')(d)
    final_output = tf.keras.layers.Dense(25, activation='linear', name='q_values')(d)
    return tf.keras.Model(inputs=[board_input, swap_input], outputs=final_output, name="training_model")

def create_inference_model():
    """ A 1-to-1 replica using standard Dense layers. """
    board_input = tf.keras.layers.Input(shape=(5, 5, 2), name='board_input')
    swap_input = tf.keras.layers.Input(shape=(1,), name='swap_input')
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='local_patterns')(board_input)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='global_patterns')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='pattern_combinations')(x)
    x_flat = tf.keras.layers.Flatten(name='flatten')(x)
    concatenated = tf.keras.layers.Concatenate(name='concatenate')([x_flat, swap_input])
    d = tf.keras.layers.Dense(256, name='decision_layer_1')(concatenated)
    d = tf.keras.layers.ReLU(name='re_lu_1')(d)
    d = tf.keras.layers.Dense(128, name='decision_layer_2')(d)
    d = tf.keras.layers.ReLU(name='re_lu_2')(d)
    final_output = tf.keras.layers.Dense(25, activation='linear', name='q_values')(d)
    return tf.keras.Model(inputs=[board_input, swap_input], outputs=final_output, name="inference_model")

# --- Main Export Logic ---

# 1. Load the original model to access its weights.
print("Step 1: Loading original model...")
# Note: Keras needs the custom layer class available to load the model structure.
loaded_model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={'NoisyFactorisedDense': NoisyFactorisedDense}
)
print("Original model loaded.")

# 2. Create the clean, structurally identical inference model.
print("\nStep 2: Creating clean inference model...")
inference_model = create_inference_model()
print("Inference model created.")

# 3. Manually and explicitly copy weights layer by layer, matching by name.
print("\nStep 3: Manually copying weights by layer name...")
for layer_loaded in loaded_model.layers:
    # Skip layers that have no weights (Input, Flatten, ReLU, etc.)
    if not layer_loaded.get_weights():
        continue

    try:
        # Find the identically named layer in our new model
        layer_inference = inference_model.get_layer(name=layer_loaded.name)

        # Special handling for converting NoisyFactorisedDense to Dense
        if isinstance(layer_loaded, NoisyFactorisedDense):
            print(f"  - Converting weights for Noisy layer: {layer_loaded.name}")
            # A Noisy layer's weights are [mu_w, mu_b, sigma_w, sigma_b].
            # A Dense layer's weights are [kernel, bias].
            # We copy the first two (mu_w, mu_b) to the Dense layer.
            mean_weights = layer_loaded.get_weights()[:2]
            layer_inference.set_weights(mean_weights)
        else:
            # For standard layers (Conv2D, Dense), copy them directly.
            print(f"  - Copying weights for standard layer: {layer_loaded.name}")
            layer_inference.set_weights(layer_loaded.get_weights())

    except ValueError:
        print(f"  - WARNING: Layer '{layer_loaded.name}' found in original model but not in inference model.")

print("Weight copy complete.")


# 4. Define the serving signature and save the INFERENCE model.
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 5, 5, 2], dtype=tf.float32, name="board_input"),
    tf.TensorSpec(shape=[None, 1],       dtype=tf.float32, name="swap_input")
])
def serve_fn(board_input, swap_input):
    return {"q_values": inference_model([board_input, swap_input], training=False)}

print(f"\nStep 4: Exporting the final INFERENCE model to {EXPORT_DIR}...")
if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

tf.saved_model.save(
    inference_model,
    EXPORT_DIR,
    signatures={"serving_default": serve_fn}
)

print("\nModel exported successfully! You may now run the Java application.")
import tensorflow as tf
import numpy as np
import os
import shutil

from tensorflow.keras.layers import Input, Dense, Conv2D, ReLU, Concatenate, Flatten
from tensorflow.keras.models import Model
from DQNNFD import NoisyFactorisedDense # Assuming DQNNFD.py is in the same directory

# --- Configuration ---
KERAS_MODEL_PATH = 'models/5x5-simple_____4.90max____4.56avg____0.00min__1750339997.keras'
EXPORT_TXT_FILENAME = 'djl_weights.txt'
EXPORT_DIR = "exported_inference_model_for_java" # New directory for SavedModel if needed

# --- Model Definitions (unchanged from your original) ---
def create_training_model():
    """Defines the original model architecture with custom Noisy layers."""
    board_input = Input(shape=(5, 5, 2), name='board_input')
    swap_input = Input(shape=(1,), name='swap_input')
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='local_patterns')(board_input)
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='global_patterns')(x)
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='pattern_combinations')(x)
    x_flat = Flatten()(x)
    concatenated = Concatenate()([x_flat, swap_input])
    d = NoisyFactorisedDense(256, name='decision_layer_1')(concatenated)
    d = ReLU()(d)
    d = NoisyFactorisedDense(128, name='decision_layer_2')(d)
    d = ReLU()(d)
    final_output = Dense(25, activation='linear', name='q_values')(d)
    return Model(inputs=[board_input, swap_input], outputs=final_output)

def create_inference_model():
    """Defines a clean model architecture with standard Dense layers for deployment."""
    board_input = Input(shape=(5, 5, 2), name='board_input')
    swap_input = Input(shape=(1,), dtype=tf.float32, name='swap_input')
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='local_patterns')(board_input)
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='global_patterns')(x)
    x = Conv2D(128, kernel_size=3, padding='same', activation='relu', name='pattern_combinations')(x)
    x_flat = Flatten()(x)
    concatenated = Concatenate()([x_flat, swap_input])
    d = Dense(256, name='decision_layer_1')(concatenated)
    d = ReLU()(d)
    d = Dense(128, name='decision_layer_2')(d)
    d = ReLU()(d)
    final_output = Dense(25, activation='linear', name='q_values')(d)
    return Model(inputs=[board_input, swap_input], outputs=final_output)

def export_weights_to_text_file():
    """
    Loads the model, extracts weights, and saves them to a plain text file.
    The format for each line will be:
    <layer_name>_<weight_type>:<shape_dim1>,<shape_dim2>,...:<value1>,<value2>,...
    """
    print("Loading original training model to extract weights...")
    training_model = create_training_model()
    # Ensure the model is built before loading weights if using custom layers
    training_model.build(input_shape=[(None, 5, 5, 2), (None, 1)])
    training_model.load_weights(KERAS_MODEL_PATH)
    print("Original training model loaded.")

    print("Creating clean inference model...")
    inference_model = create_inference_model()
    # Ensure inference model is built
    inference_model.build(input_shape=[(None, 5, 5, 2), (None, 1)])
    print("Inference model created.")

    print("Copying weights to inference model...")
    for layer_train in training_model.layers:
        if not layer_train.get_weights():
            continue
        try:
            layer_inference = inference_model.get_layer(name=layer_train.name)
            if isinstance(layer_train, NoisyFactorisedDense):
                layer_inference.set_weights(layer_train.get_weights()[:2])
            else:
                layer_inference.set_weights(layer_train.get_weights())
        except ValueError:
            # Handle nested layers if your inference model uses Sequential inside
            try:
                # Check if it's a layer from the conv_base sequential model
                if 'conv_base' in inference_model.layers[0].name and layer_train.name in [l.name for l in inference_model.get_layer('conv_base').layers]:
                    inference_model.get_layer('conv_base').get_layer(layer_train.name).set_weights(layer_train.get_weights())
                # Check if it's a layer from the decision_head sequential model
                elif 'decision_head' in inference_model.layers[1].name and layer_train.name in [l.name for l in inference_model.get_layer('decision_head').layers]:
                    # This part needs to be adapted based on the exact structure if you used Sequential
                    # For a simple Functional API model, this 'try-except' block might not be strictly needed
                    # if layer names are unique across the model.
                    if isinstance(layer_train, NoisyFactorisedDense):
                        inference_model.get_layer('decision_head').get_layer(layer_train.name).set_weights(layer_train.get_weights()[:2])
                    else:
                        inference_model.get_layer('decision_head').get_layer(layer_train.name).set_weights(layer_train.get_weights())
                else:
                    print(f"  - WARNING: Layer '{layer_train.name}' not found in inference model or its sub-models during direct copy.")
            except AttributeError: # Happens if get_layer doesn't return a Sequential model
                print(f"  - WARNING: Layer '{layer_train.name}' could not be copied. It might be a functional API intermediate tensor or not directly transferable.")

    print("Weight copy complete.")
    
    print(f"\nExtracting weights and saving to '{EXPORT_TXT_FILENAME}'...")
    with open(EXPORT_TXT_FILENAME, 'w') as f:
        for layer in inference_model.layers:
            if not layer.get_weights():
                continue
            
            # Map Keras layer names to DJL-style names
            param_name_base = layer.name.replace("decision_layer_", "decisionLayer")\
                                       .replace("q_values", "qValuesLayer")
            
            weights = layer.get_weights()
            
            # Save kernel weights
            kernel_name = f"{param_name_base}_weight"
            kernel_data = weights[0].flatten().tolist()
            kernel_shape = ",".join(map(str, weights[0].shape))
            f.write(f"{kernel_name}:{kernel_shape}:{','.join(map(str, kernel_data))}\n")
            
            # Save bias weights
            bias_name = f"{param_name_base}_bias"
            bias_data = weights[1].flatten().tolist()
            bias_shape = ",".join(map(str, weights[1].shape))
            f.write(f"{bias_name}:{bias_shape}:{','.join(map(str, bias_data))}\n")
            
    print("\nSUCCESS: Weight text file has been created.")

    # Optional: Also save the TF SavedModel for completeness, though not used by Java directly
    print(f"\nExporting the final INFERENCE model as TensorFlow SavedModel to {EXPORT_DIR}...")
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 5, 5, 2], dtype=tf.float32, name="board_input"),
        tf.TensorSpec(shape=[None, 1],       dtype=tf.float32, name="swap_input")
    ])
    def serve_fn(board_input, swap_input):
        return {"q_values": inference_model([board_input, swap_input], training=False)}

    tf.saved_model.save(
        inference_model,
        EXPORT_DIR,
        signatures={"serving_default": serve_fn}
    )
    print("TensorFlow SavedModel exported successfully!")


if __name__ == '__main__':
    export_weights_to_text_file()

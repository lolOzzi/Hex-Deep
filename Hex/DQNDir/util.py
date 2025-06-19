import numpy as np
import tensorflow as tf
from DQNNFD import *

def reset_noise_in_model(agent, new_sigma_init=0.5):
    """
    Finds NoisyFactorisedDense layers and resets their sigma weights.
    """
    print("Resetting noise parameters in the model...")
    for layer in agent.model.layers:
        if isinstance(layer, NoisyFactorisedDense):
            print(f"  - Resetting noise for layer: {layer.name}")

            # Get the original initializers and shapes
            in_features = layer.in_features
            
            # Re-initialize sigma weights
            new_sigma_w = tf.keras.initializers.Constant(new_sigma_init / tf.sqrt(float(in_features)))(shape=layer.sigma_w.shape)
            new_sigma_b = tf.keras.initializers.Constant(new_sigma_init / tf.sqrt(float(in_features)))(shape=layer.sigma_b.shape)
            
            # Set the new weights in the layer
            layer.sigma_w.assign(new_sigma_w)
            layer.sigma_b.assign(new_sigma_b)
            
    agent.target_model.set_weights(agent.model.get_weights())
    print("Noise parameter reset complete.")


def loadModel(agent, model_file_path):
    if model_file_path:
        print(f"Loading model from: {model_file_path}")
        try:

            loaded_model = tf.keras.models.load_model(
                model_file_path,
                custom_objects={'NoisyFactorisedDense': NoisyFactorisedDense}
            )
            
            agent.model.set_weights(loaded_model.get_weights())
            agent.target_model.set_weights(agent.model.get_weights())
            
            print("Model loaded and recompiled successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch (or original compile learning rate).")
            agent.target_model.set_weights(agent.model.get_weights())
    else:
        print("No model file found, starting from scratch.")
        agent.target_model.set_weights(agent.model.get_weights())
def loadPartialModel(agent, model_file):
    agent.load_partial_weights(model_file)

def exportModel(agent):
    dummy_board = np.zeros((1, 5, 5, 2), dtype=np.float32)
    dummy_swap  = np.zeros((1, 1), dtype=np.float32)
    _ = agent.model([dummy_board, dummy_swap], training=False)
    @tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 5, 5, 2], dtype=tf.float32, name="board_input"),
    tf.TensorSpec(shape=[None, 1],    dtype=tf.float32, name="swap_input")
    ])
    def serve_fn(board_input, swap_input):
        # Run inference (no training) to get Q-values
        outputs = agent.model([board_input, swap_input], training=False)
        # Return a dict so the signature names the output tensor
        return {"q_values": outputs}

    # Directory where to save; choose a fresh directory
    export_dir = "exported_simple_hex_dqn"

    # Save the model with this signature under the default serving key
    tf.saved_model.save(agent.model, export_dir, signatures={"serving_default": serve_fn})
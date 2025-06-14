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
            agent.model = tf.keras.models.load_model(
                model_file_path,
                custom_objects={'NoisyFactorisedDense': NoisyFactorisedDense}
            )
            agent.target_model.set_weights(agent.model.get_weights())
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
    else:
        print("No model file found, starting from scratch.")
def loadPartialModel(model_file):
    agent.load_partial_weights(model_file)
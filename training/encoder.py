import numpy as np
import tensorflow as tf


#----------------------------------------------------------------------------
# Encoder network.
# Extract the feature of content and style image
# Use VGG19 network to extract features.

def VGG_Encoder(
    images_in,                                  # Input Images
    image_shape     = [3, 256, 256],            # Input Image shape
    grid_batch      = 480,                      # Batch size
    latent_size     = 512,                      # Latent code size
    **kwargs):

    # For test
    return np.random.randn(grid_batch, latent_size)
    return None
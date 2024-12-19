import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import os

output_dir = 'GAN_synthetic_Sample_Output_'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the generator TFLite model
generator_model_path = 'Fingerprint_code/GAN_Synthetic_code/ckpt_/generator_model.tflite'
generator_interpreter = tf.lite.Interpreter(model_path=generator_model_path)
generator_interpreter.allocate_tensors()

# Get input and output details
generator_input_details = generator_interpreter.get_input_details()
generator_output_details = generator_interpreter.get_output_details()

# Generate and save 100 grayscale images
for i in range(10):
    # Generate a random latent input tensor
    latent_dim = 100 # specify the latent dimension of the generator
    latent_input = np.random.randn(1, latent_dim).astype(np.float32)

    # Set input tensor to the generator
    generator_interpreter.set_tensor(generator_input_details[0]['index'], latent_input)

    # Invoke the generator
    generator_interpreter.invoke()

    # Get the generated image tensor
    generated_image = generator_interpreter.get_tensor(generator_output_details[0]['index'])

    # Process the generated image as needed (e.g., convert to an image format or perform post-processing)

    # Convert the generated image tensor to an image format
    generated_image = np.squeeze(generated_image, axis=0)
    generated_image = (generated_image + 1) / 2  # Convert values from [-1, 1] to [0, 1]

    # Convert the image to grayscale
    generated_image_gray = generated_image[..., 0] * 0.2989 + generated_image[..., 0] * 0.5870 + generated_image[..., 0] * 0.1140

    # Create an Image object from the grayscale image array
    generated_image_gray = Image.fromarray(np.uint8(generated_image_gray * 255), 'L')

    # Save the image with a unique filename
    filename = f'generated_image_20240401_{i+1}.jpg'
    file_path = os.path.join(output_dir, filename)
    generated_image_gray.save(file_path)

    print(f'Saved image: {file_path}')

# Generating Fake Fingerprint Images Using a GAN Model

This project demonstrates how to generate synthetic fingerprint images using a pre-trained GAN (Generative Adversarial Network) generator model. The generated images are saved as grayscale images in a specified output directory.

## Prerequisites

1. **Python Libraries**:
    - TensorFlow: For loading and invoking the TFLite generator model.
    - NumPy: For generating random latent vectors and processing images.
    - Matplotlib: For visualization (optional).
    - PIL (Python Imaging Library): For saving the generated images.
    - OS: For handling directories and file paths.

2. **Generator Model**:
    - A pre-trained TFLite generator model file named `generator_model.tflite`.

3. **Directory Structure**:
    - Ensure that the model file is located in the directory `Fingerprint_code/GAN_Synthetic_code/ckpt_/`.

4. **Output Directory**:
    - The generated images will be saved in a directory named `GAN_synthetic_Sample_Output_`. If the directory does not exist, it will be created automatically.

## How It Works

1. The script loads a pre-trained generator model in TensorFlow Lite format using the TensorFlow Lite interpreter.
2. A latent vector (random noise) is generated as input for the GAN.
3. The generator model processes the latent vector and produces synthetic fingerprint images.
4. The generated images are post-processed:
    - Values are scaled from the range [-1, 1] to [0, 1].
    - Images are converted to grayscale.
    - Images are saved in the specified output directory with unique filenames.
5. The process is repeated to generate multiple images.

## Code Overview

### 1. Setting Up the Output Directory
```python
output_dir = 'GAN_synthetic_Sample_Output_'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```
- Creates the output directory if it does not exist.

### 2. Loading the Generator Model
```python
generator_model_path = 'Fingerprint_code/GAN_Synthetic_code/ckpt_/generator_model.tflite'
generator_interpreter = tf.lite.Interpreter(model_path=generator_model_path)
generator_interpreter.allocate_tensors()
```
- Loads the pre-trained generator model in TensorFlow Lite format.

### 3. Generating Synthetic Fingerprint Images
- Random latent vectors are generated as input to the GAN:
```python
latent_dim = 100
latent_input = np.random.randn(1, latent_dim).astype(np.float32)
```
- The latent vector is passed to the model, and the output is post-processed:
```python
# Set input tensor
generator_interpreter.set_tensor(generator_input_details[0]['index'], latent_input)

# Invoke the generator
generator_interpreter.invoke()

# Get the generated image
generated_image = generator_interpreter.get_tensor(generator_output_details[0]['index'])

# Post-process the image
generated_image = (generated_image + 1) / 2
```
- The generated image is converted to grayscale and saved:
```python
generated_image_gray = Image.fromarray(np.uint8(generated_image * 255), 'L')
file_path = os.path.join(output_dir, filename)
generated_image_gray.save(file_path)
```

### 4. Saving the Images
- Each generated image is saved with a unique filename:
```python
filename = f'generated_image_20240401_{i+1}.jpg'
```

## Usage

1. Run the script using Python:
```bash
python Main.py
```

2. The generated images will be saved in the `GAN_synthetic_Sample_Output_` directory.

## Output
- The script generates 10 synthetic fingerprint images and saves them as grayscale `.jpg` files.
- Example filenames:
  - `generated_image_20240401_1.jpg`
  - `generated_image_20240401_2.jpg`

## Notes
- Ensure that the generator model file exists at the specified path.
- Modify the `latent_dim` variable if your generator requires a different latent dimension.
- Adjust the number of generated images by changing the loop range.

## Acknowledgments
This project uses a GAN generator model for synthetic fingerprint image generation. It demonstrates the potential of GANs in producing high-quality synthetic data for testing and research purposes.


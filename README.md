# MLFlow Model Training Package

This package provides a general-purpose deep learning model training function, integrated with MLflow for logging and tracking experiments. The model supports TensorFlow-based training and allows easy configuration of various hyperparameters such as input shapes, dense layers, and callbacks. Additionally, it logs important metrics like loss, learning rate, and model performance using MLflow.

## Features
- **MLflow Integration**: Automatically tracks and logs experiments, parameters, and metrics.
- **Flexible Model Design**: Easily configure the model architecture (input layers, dense layers, and output layer).
- **Callbacks**: Provides custom callbacks for logging metrics and learning rate to MLflow.
- **Device Selection**: Choose whether to train on CPU or GPU.
- **Default Parameters**: Provides sensible defaults for common use cases, while allowing full customization.

## Installation

To install the package, use `pip`:

```bash
pip install gtek_mlflow
```

## Example Usage

Below is an example of how to use the `train_model` function from the package with a simple TensorFlow data generator.

```python
import tensorflow as tf
from mlflowlib import training

# Define a sample data generator
def generate_data():
    # Example data generator for simulation
    for _ in range(100):
        x = tf.random.normal((8, 670, 1413, 3))  # Input 1 (image data)
        y = tf.random.normal((100, 24))          # Input 2 (wind data)
        z = tf.random.normal((1, 2400))          # Output
        yield [x, y], z

# Prepare training dataset
train_dataset = tf.data.Dataset.from_generator(
    generate_data,
    output_signature=(
        (tf.TensorSpec(shape=(8, 670, 1413, 3), dtype=tf.float32),  # Input 1 (image data)
         tf.TensorSpec(shape=(100, 24), dtype=tf.float32)),         # Input 2 (wind data)
        tf.TensorSpec(shape=(1, 2400), dtype=tf.float32)            # Output
    )
).batch(32)

# Prepare a test dataset
test_dataset = train_dataset.take(10)

# Train the model using the train_model function
training(
    experiment_name='Satellite_Wind_Prediction',
    train_generator=train_dataset,
    test_generator=test_dataset,
    epochs=5,  # Default is 100, but set to 5 here for a shorter run
    batch_size=32,
    device='/CPU:0',  # Specify to run on CPU (can switch to GPU if available)
    optimizer='adam',
    loss_function='mse',
    metrics=['mae', 'mse']
)
```
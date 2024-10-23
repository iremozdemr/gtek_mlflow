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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, Reshape, Concatenate
from mlflowlib import train_model  

# Function to create a sample model
def create_model(input_image_shape=(8, 670, 1413, 3), turbine_total_count=100):
    # Input layers for the model
    input_images = Input(shape=input_image_shape, name='image_input')
    input_wind_speeds = Input(shape=(turbine_total_count, 24), name='wind_speed_input')

    # CNN layers
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(input_images)
    x = Conv3D(32, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(x)
    x = Conv3D(16, kernel_size=(3, 3, 3), strides=(4, 4, 4), activation='relu', padding='same')(x)
    x = Flatten()(x)

    # Flatten the wind speed input
    flattened_wind_speeds = Flatten()(input_wind_speeds)

    # Concatenate image and wind speed inputs
    combined = Concatenate()([x, flattened_wind_speeds])

    # Fully connected layers
    combined = Dense(128, activation='relu')(combined)
    combined = Dense(64, activation='relu')(combined)

    # Output layer
    output = Dense(turbine_total_count * 24, activation='linear')(combined)
    output = Reshape((turbine_total_count, 24))(output)

    # Create the model
    model = Model(inputs=[input_images, input_wind_speeds], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

# Data generator function (for generating example data)
def generate_data():
    for _ in range(100):
        x = tf.random.normal((8, 670, 1413, 3))  # Image data
        y = tf.random.normal((100, 24))          # Wind speed data
        z = tf.random.normal((100, 24))          # Output
        yield [x, y], z

# Prepare training and test datasets
train_dataset = tf.data.Dataset.from_generator(
    generate_data,
    output_signature=(
        (tf.TensorSpec(shape=(8, 670, 1413, 3), dtype=tf.float32),  # Image data
         tf.TensorSpec(shape=(100, 24), dtype=tf.float32)),         # Wind speed data
        tf.TensorSpec(shape=(100, 24), dtype=tf.float32)            # Output
    )
).batch(32)

test_dataset = train_dataset.take(10)  # Take 10 batches as test data

# Create the model
model = create_model()

# Call the train_model function to train the model
training.train_model(
    run_name="example_run",
    tracking_uri="",
    experiment_name="Wind_Turbine_Prediction",
    batch_size=32,
    epochs=5,
    device='/CPU:0',  # Use '/GPU:0' if you have a GPU available
    model=model
)
```

```python
from mlflowlib import train_model

training.train_model(
    run_name="example_run",
    tracking_uri="",
    experiment_name="Wind_Turbine_Prediction",
    batch_size=32,
    epochs=5,
    device='/CPU:0',  # Use '/GPU:0' if you have a GPU available
    model=model
)
```
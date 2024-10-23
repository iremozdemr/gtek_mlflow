#Delete all files in the dist folder
#Update the version number in the setup.py file
#Re-create the wheels:
#python3 setup.py sdist bdist_wheel
#Re-upload the new files:
#twine upload dist/*

import mlflow
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Flatten, Concatenate, Dense, Reshape
from tensorflow.keras.callbacks import Callback
import time

# Callback to log learning rate
class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs = logs or {}
        mlflow.log_metric('learning_rate', float(lr.numpy()), step=epoch)

# Callback to log loss and errors
class LossAndErrorPrintingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)

# Custom callback to track and print progress
class ProgressLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.params['epochs']} started.")
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        print(f"Epoch {epoch+1}/{self.params['epochs']} completed in {duration:.2f} seconds.")
        print(f"Metrics: {logs}")
        progress = ((epoch+1) / self.params['epochs']) * 100
        print(f"Progress: {progress:.2f}% completed.")

    def on_batch_end(self, batch, logs=None):
        progress = ((batch+1) / self.params['steps']) * 100
        print(f"Batch {batch+1}/{self.params['steps']} - {progress:.2f}% completed.")

# General model training function
def train_model(
    run_name,
    tracking_uri="http://your_mlflow_server_uri:5000",
    experiment_name='using mlflowlib',
    batch_size=32,
    epochs=7,
    device='/CPU:0',
    input_image_shape=(8, 670, 1413, 3),
    turbine_total_count=100,
    train_generator=None,
    test_generator=None,
    callbacks=None,
    optimizer='adam',
    loss_function='mse',
    metrics=['mae', 'mape', 'mse']
):
    """
    Function for training a model using MLflow and TensorFlow. Performs the entire
    model training and logging with a single function call.

    Args:
        run_name (str): Name of the run in MLflow.
        tracking_uri (str, optional): MLflow tracking URI. Default 'http://your_mlflow_server_uri:5000'.
        experiment_name (str, optional): Name of the experiment in MLflow. Default 'using mlflowlib'.
        batch_size (int, optional): Batch size for training. Default 32.
        epochs (int, optional): Number of epochs for training. Default 7.
        device (str, optional): Device to be used for training. Default 'CPU:0'.
        input_image_shape (tuple, optional): Shape of the input image. Default (8, 670, 1413, 3).
        turbine_total_count (int, optional): Total number of turbines. Default 100.
        train_generator (tf.data.Dataset): Training data generator. Required.
        test_generator (tf.data.Dataset): Test data generator. Required.
        callbacks (list, optional): List of custom callbacks. Default callbacks: `LearningRateLogger`, `LossAndErrorPrintingCallback`, and `ProgressLogger`.
        optimizer (str, optional): Optimizer to be used. Default 'adam'.
        loss_function (str, optional): Loss function. Default 'mse'.
        metrics (list, optional): Metrics to be used during training. Default ['mae', 'mape', 'mse'].
    
    Returns:
        None
    """

    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        with tf.device(device):
            # Create input layers
            input_images = Input(shape=input_image_shape, name='image_input')

            x = Conv3D(64, kernel_size=(3, 3, 3), strides=(4,4,4), activation='relu', padding='same')(input_images)
            x = Conv3D(32, kernel_size=(3, 3, 3), strides=(4,4,4), activation='relu', padding='same')(x)
            x = Conv3D(16, kernel_size=(3, 3, 3), strides=(4,4,4), activation='relu', padding='same')(x)
            x = Flatten()(x)

            input_wind_speeds = Input(shape=(turbine_total_count, 24), name='wind_speed_input')
            flattened_wind_speeds = Flatten()(input_wind_speeds)

            combined = Concatenate()([x, flattened_wind_speeds])

            combined = Dense(128, activation='relu')(combined)
            combined = Dense(64, activation='relu')(combined)

            output = Dense(turbine_total_count * 24, activation='linear')(combined)
            output = Reshape((turbine_total_count, 24))(output)

            model = Model(inputs=[input_images, input_wind_speeds], outputs=output)

            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            # Default Callbacks if none provided
            if callbacks is None:
                lr_logger_callback = LearningRateLogger()
                get_metrics_callback = LossAndErrorPrintingCallback()
                progress_logger_callback = ProgressLogger()
                callbacks = [lr_logger_callback, get_metrics_callback, progress_logger_callback]

            # Train the model
            model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=len(train_generator),
                validation_data=test_generator,
                callbacks=callbacks
            )

            # Log the model to MLflow
            mlflow.keras.log_model(model, "model")
import mlflow
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Flatten, Concatenate, Dense, Reshape
from tensorflow.keras.callbacks import Callback

# MLflow kurulumu
def setup_mlflow(tracking_uri, experiment_name, autolog=False):
    """
    MLflow'un takip URI'sini ve deney adini yapilandirir. Istege bagli olarak
    MLflow'un TensorFlow autolog fonksiyonunu etkinlestirebilir.

    Args:
        tracking_uri (str): MLflow takip URI'si. MLflow sunucusunun adresi.
        experiment_name (str): Deney adi. MLflow'da deneylerin kaydedilecegi deney adi.
        autolog (bool, optional): Eger True olarak ayarlanirsa, MLflow otomatik loglama yapar.
            Varsayilan deger False.
    
    Returns:
        None
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if autolog:
        mlflow.tensorflow.autolog(log_models=False)

# Ogrenme oranini loglayan callback
class LearningRateLogger(Callback):
    """
    Model egitiminde her epoch'un sonunda ogrenme oranini MLflow'a loglayan 
    bir callback sinifi. Ogrenme orani, her epoch sonunda MLflow'da 
    'learning_rate' metrigi olarak kaydedilir.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Her epoch sonunda cagrilan fonksiyon. Modelin ogrenme oranini loglar.

        Args:
            epoch (int): O anki epoch numarasi.
            logs (dict, optional): Egitim sirasinda olusan metrik degerleri.
        
        Returns:
            None
        """
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs = logs or {}
        mlflow.log_metric('learning_rate', float(lr.numpy()), step=epoch)

# Kayip ve hatalari loglayan callback
class LossAndErrorPrintingCallback(Callback):
    """
    Model egitiminde her epoch sonunda kayip ve hatalari MLflow'a loglayan 
    bir callback sinifi. Modelin kayip (loss) ve hata (metrics) metriklerini 
    MLflow'da kaydeder.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Her epoch sonunda cagrilan fonksiyon. Epoch sirasinda olusan metrikleri loglar.

        Args:
            epoch (int): O anki epoch numarasi.
            logs (dict, optional): Egitim sirasinda olusan kayip ve hata metrikleri.
        
        Returns:
            None
        """
        mlflow.log_metrics(logs, step=epoch)

# Genel model egitim fonksiyonu
def train_model(
    experiment_name,  # Varsayilan deney adi
    tracking_uri="http://your_mlflow_server_uri:5000",  # Varsayilan olarak yerel bir MLflow sunucusu
    batch_size=32,  # Varsayilan batch boyutu
    epochs=100,  # Varsayilan epoch sayisi
    device='/CPU:0',  # Varsayilan olarak CPU'da calistirilir
    input_shapes=[(8, 670, 1413, 3), (100, 24)],  # Varsayilan girdi sekilleri
    model_layers=[128, 64],  # Varsayilan Dense katman sayilari
    output_units=100 * 24,  # Varsayilan cikis birim sayisi
    train_generator=None,  # Veri jeneratörü zorunlu
    test_generator=None,  # Test jeneratörü zorunlu
    optimizer='adam',  # Varsayilan optimizer
    loss_function='mse',  # Varsayilan kayip fonksiyonu
    metrics=['mae', 'mape', 'mse'],  # Varsayilan metrikler
    callbacks=None,  # Varsayilan callback'ler None
    autolog=False  # MLflow autolog varsayilan olarak False
):
    """
    Modelin egitilmesini ve MLflow ile loglanmasini saglayan fonksiyon.

    Args:
        tracking_uri (str, optional): MLflow takip URI'si. Varsayilan 'http://127.0.0.1:5000'.
        experiment_name (str, optional): Deney adi. 
        batch_size (int, optional): Egitim sirasinda kullanilacak batch boyutu. Varsayilan 32.
        epochs (int, optional): Egitimde kac epoch yapilacagini belirler. Varsayilan 100.
        device (str, optional): Modelin hangi cihazda egitilecegini belirtir. Varsayilan 'CPU:0'.
        input_shapes (list, optional): Girdi katmanlarinin sekillerini iceren liste. Varsayilan [(8, 670, 1413, 3), (100, 24)].
        model_layers (list, optional): Dense katmanlarindaki birim sayilarini iceren liste. Varsayilan [128, 64].
        output_units (int, optional): Cikis katmaninin birim sayisi. Varsayilan 100 * 24.
        train_generator (tf.data.Dataset): Egitim verileri icin veri jeneratoru. Zorunludur.
        test_generator (tf.data.Dataset): Test verileri icin veri jeneratoru. Zorunludur.
        optimizer (str, optional): Kullanilacak optimizer. Varsayilan 'adam'.
        loss_function (str, optional): Kayip fonksiyonu. Varsayilan 'mse'.
        metrics (list, optional): Egitim sirasinda loglanacak metrikler. Varsayilan ['mae', 'mape', 'mse'].
        callbacks (list, optional): Kullanici tarafindan belirlenen callback fonksiyonlari.
            Eger belirtilmezse, varsayilan olarak 'LearningRateLogger' ve 'LossAndErrorPrintingCallback' kullanilir.
        autolog (bool, optional): Eger True ise MLflow'un TensorFlow autolog ozelligi kullanilir. Varsayilan False.
    
    Returns:
        None
    """
    
    # MLflow ayarlarini baslat
    setup_mlflow(tracking_uri, experiment_name, autolog)

    with mlflow.start_run(run_name=f'ML_Model_Training') as run:
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        with tf.device(device):
            # Model girdilerinin dinamik olarak yapilandirilmasi
            inputs = []
            flattened_inputs = []

            for shape in input_shapes:
                input_layer = Input(shape=shape)
                inputs.append(input_layer)
                flattened_inputs.append(Flatten()(input_layer))

            # Katmanlarin birlestirilmesi
            combined = Concatenate()(flattened_inputs)

            # Modelde belirtilen Dense katmanlari olusturma
            for units in model_layers:
                combined = Dense(units, activation='relu')(combined)

            # Sonuc katmani (cikis)
            output = Dense(units=output_units, activation='linear')(combined)
            output = Reshape((1, output_units))(output)  # Cikis boyutu ihtiyaca gore duzenlenebilir

            # Modelin derlenmesi
            model = Model(inputs=inputs, outputs=output)
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            # Varsayilan callback'ler (Geri cagirim fonksiyonlari)
            if callbacks is None:
                lr_logger_callback = LearningRateLogger()
                get_metrics_callback = LossAndErrorPrintingCallback()
                callbacks = [lr_logger_callback, get_metrics_callback]

            # Modelin egitilmesi
            history = model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=len(train_generator),
                validation_data=test_generator,
                callbacks=callbacks
            )

            # Modeli MLflow'a kaydet
            mlflow.keras.log_model(model, "model")
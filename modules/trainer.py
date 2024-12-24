
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras.utils import plot_model
import os

# Definisikan nama label
LABEL_KEY = "target"

# Daftar fitur numerik dan kategorikal
NUMERIC_FEATURES = ['age', 'ca', 'chol', 'oldpeak', 'thalach', 'trestbps']
CATEGORICAL_FEATURES = ['cp', 'exang', 'fbs', 'restecg', 'sex', 'slope', 'thal']

def transformed_name(key):
    """Menambahkan suffix '_xf' pada nama fitur yang telah ditransformasi"""
    return key + "_xf"

# Fungsi untuk membaca data yang telah di-compress
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Fungsi input untuk mempersiapkan dataset
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    # Mendapatkan feature_spec untuk fitur yang sudah ditransformasi
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Membaca data dalam bentuk batch
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key='target'  # Menggunakan 'target' sebagai label key yang benar
    )

    # Fungsi untuk format data (menyesuaikan label dan fitur)
    def format_data(features, labels):
        labels = tf.reshape(labels, [-1, 1])  # Bentuk label sesuai dengan output
        return features, labels

    return dataset.map(format_data)


# Membangun model
def model_builder():
    inputs = []

    # Menambahkan input layer untuk setiap fitur
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        inputs.append(tf.keras.Input(shape=(1,), name=transformed_name(feature)))

    # Menggabungkan input layer
    x = tf.keras.layers.Concatenate()(inputs)

    # Hidden layers
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Output layer untuk klasifikasi
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Membuat model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Menyusun model dengan optimizer dan loss function
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model

# Fungsi untuk menyajikan TF examples
def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn

# Fungsi untuk mendapatkan signature dari fitur transformasi
def _get_transform_features_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn

# Fungsi utama untuk menjalankan pelatihan
def run_fn(fn_args: FnArgs):
    # Menginisialisasi tf_transform_output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Membaca dataset pelatihan dan evaluasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=20)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)

    # Membangun model
    model = model_builder()

    # Melatih model
    model.fit(train_dataset, epochs=20, validation_data=eval_dataset)

    # Menyimpan model dengan signatures untuk serving
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }

    # Menyimpan model yang telah dilatih
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)

    plot_model(
        model, 
        to_file='images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )

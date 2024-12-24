
# Import library
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "target"

# List of feature names and types
NUMERIC_FEATURES = ['age', 'ca', 'chol', 'oldpeak', 'thalach', 'trestbps']
CATEGORICAL_FEATURES = ['cp', 'exang', 'fbs', 'restecg', 'sex', 'slope', 'thal']

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: dictionary of raw input features.

    Returns:
        outputs: dictionary of transformed features.
    """
    outputs = {}

    # Filter out rows with invalid values for 'ca' and 'thal'
    valid_rows = tf.logical_and(
        tf.not_equal(inputs['ca'], 4),  # Exclude rows where `ca` is 4
        tf.not_equal(inputs['thal'], 0)  # Exclude rows where `thal` is 0
    )

    # Apply the valid row mask to all inputs
    filtered_inputs = {key: tf.boolean_mask(inputs[key], valid_rows) for key in inputs}

    # Normalize numeric features
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(filtered_inputs[feature])

    # Label encode categorical features
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(filtered_inputs[feature])

    # Include the label
    outputs[LABEL_KEY] = filtered_inputs[LABEL_KEY]

    return outputs

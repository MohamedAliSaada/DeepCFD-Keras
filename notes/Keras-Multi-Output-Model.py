import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ===== 1. Create dummy data =====
num_samples = 1000
num_features = 10

X = np.random.rand(num_samples, num_features)

# Output 1: Regression target (e.g., age)
y_regression = np.random.rand(num_samples, 1)

# Output 2: Classification target (e.g., 3 classes)
y_class = np.random.randint(0, 3, size=(num_samples,))
y_class = to_categorical(y_class, num_classes=3)

# ===== 2. Define the model =====
input_layer = Input(shape=(num_features,), name='input')

# Shared dense layer
shared = Dense(64, activation='relu')(input_layer)

# Output 1: Regression
regression_output = Dense(1, name='regression')(shared)

# Output 2: Classification
classification_output = Dense(3, activation='softmax', name='classification')(shared)

# Create the multi-output model
model = Model(inputs=input_layer, outputs=[regression_output, classification_output])

# ===== 3. Compile the model with weighted losses =====
model.compile(
    optimizer='adam',
    loss={
        'regression': 'mse',
        'classification': 'categorical_crossentropy'
    },
    loss_weights={
        'regression': 0.4,           # Give less weight to regression loss
        'classification': 1.0        # More importance to classification
    },
    metrics={
        'regression': 'mae',
        'classification': 'accuracy'
    }
)

# ===== 4. Train the model =====
history = model.fit(
    X,
    {'regression': y_regression, 'classification': y_class},
    epochs=10,
    validation_split=0.2,
    batch_size=32
)

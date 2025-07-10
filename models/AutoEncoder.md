# Autoencoder with 1 Decoder as Keras Custom Layer (double checked)

This repository provides a custom **AutoEncoder** implementation in Keras using functional encoder and decoder blocks. The design allows high flexibility, supporting batch normalization, custom activations, kernel size validation, and adjustable filters.

---

## 🔧 Features

* Custom encoder/decoder construction using sequential blocks
* Supports `Conv2D` for encoding and `Conv2DTranspose` for decoding
* Batch Normalization toggle
* Configurable activation functions
* Kernel size check for symmetry (only odd dimensions allowed)
* Easy inspection of model architecture using `My_layers_is()` method

---

## 📦 Installation

No installation needed if you use this in a Keras/TensorFlow project. Just include `autoencoder.py` in your working directory.

---

## 🚀 Usage

### 1. Import the Layer

```python
from autoencoder import AutoEncoder
```

### 2. Create an AutoEncoder instance

```python
model = AutoEncoder(
    filters=[16, 32, 64],            # list of integers for encoder/decoder channels
    kernel_size=(3, 3),              # only odd numbers allowed: (3,3), (5,5), etc.
    batch_norm=True,                # apply batch normalization if True
    activation="relu",             # activation after each layer: 'relu', 'tanh', 'sigmoid', etc.
    final_activation="sigmoid",    # final decoder activation: e.g. None, 'sigmoid', 'tanh'
    out_channels=1                  # number of output channels
)
```

### 3. Build and Use the Model

```python
import tensorflow as tf

input_tensor = tf.keras.Input(shape=(128, 128, 1))
output_tensor = model(input_tensor)
full_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
```

### 4. Compile and Train

```python
full_model.compile(optimizer='adam', loss='mse')
# dummy example
# full_model.fit(x_train, x_train, epochs=10, batch_size=32)
```

### 5. Inspect the Layers

```python
model.My_layers_is()
```

Example Output:

```
Encoder layers:
  Block 1:
    - Conv2D
    - BatchNormalization
    - Activation
  Block 2:
    - Conv2D
    - BatchNormalization
    - Activation
  Block 3:
    - Conv2D
    - BatchNormalization
    - Activation
Decoder layers:
  Block 1:
    - Conv2DTranspose
    - BatchNormalization
    - Activation
  Block 2:
    - Conv2DTranspose
    - BatchNormalization
    - Activation
  Block 3:
    - Conv2DTranspose
    - BatchNormalization
    - Activation
  Block 4:
    - Conv2DTranspose
    - Activation
```

---

## 🧪 All Parameters

| Parameter          | Type          | Description                                                  |
| ------------------ | ------------- | ------------------------------------------------------------ |
| `filters`          | list of int   | Filters to use in encoder (will reverse in decoder)          |
| `kernel_size`      | tuple of ints | Must be odd values only, e.g. (3,3), (5,5)                   |
| `batch_norm`       | bool          | Whether to apply BatchNormalization                          |
| `activation`       | str or None   | Activation function to use (e.g., 'relu', 'tanh', 'sigmoid') |
| `final_activation` | str or None   | Final activation for decoder output                          |
| `out_channels`     | int           | Number of output channels, e.g., 1 for grayscale image       |

---

## 📌 Notes

* This class inherits from `tf.keras.layers.Layer`, so it can be used as part of larger models.
* To use it as a full standalone model, wrap it inside a `tf.keras.Model` with Input/Output layers.
* The encoder and decoder blocks are dynamically built from your configuration.

---

## ✅ License

MIT License

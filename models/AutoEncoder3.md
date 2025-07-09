# Autoencoder with 3 Decoders as Keras Custom Layer

This repository provides a custom **AutoEncoder3** implementation in Keras using functional encoder and decoder blocks. The architecture extends the traditional autoencoder to support **multiple decoders** (one for each output channel), which is particularly useful for multitarget regression or multi-output tasks such as fluid simulation, optical flow, or medical image reconstruction.

---

## 🔧 Features

* Multiple decoder branches for independent outputs (e.g., Ux, Uy, p)
* Custom encoder/decoder construction using sequential blocks
* Supports `Conv2D` for encoding and `Conv2DTranspose` for decoding
* Batch Normalization toggle
* Configurable activation functions
* Kernel size validation (only odd dimensions allowed)
* Layer inspection via `My_layers_is()` method

---

## 📦 Installation

No special installation required. Just ensure this file is accessible within your TensorFlow project directory.

---

## 🚀 Usage

### 1. Import the Layer

```python
from autoencoder import AutoEncoder3
```

### 2. Create an AutoEncoder3 instance

```python
model = AutoEncoder3(
    filters=[16, 32, 64],         # encoder/decoder filters
    kernel_size=(3, 3),           # must be odd values
    batch_norm=True,              # toggle for BatchNormalization
    activation="relu",           # activation function for hidden layers
    final_activation="sigmoid",  # output layer activation
    out_channels=3                # number of independent decoders/outputs
)
```

### 3. Build and Use the Model

```python
import tensorflow as tf

input_tensor = tf.keras.Input(shape=(128, 128, 1))
output_tensor = model(input_tensor)
full_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
```

**Expected Output Shape:**
If `input_tensor` has shape `(128, 128, 1)` and `out_channels=3`, the final output will be:

```python
output_tensor.shape == (128, 128, 3)
```

Each output channel corresponds to a separate decoder branch (e.g., Ux, Uy, p).

### 4. Compile and Train

```python
full_model.compile(optimizer='adam', loss='mse')
# full_model.fit(x_train, y_train, epochs=10, batch_size=32)
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
  paths 1:
    blocks 1:
      - Conv2DTranspose
      - BatchNormalization
      - Activation
    blocks 2:
      - Conv2DTranspose
      - BatchNormalization
      - Activation
    blocks 3:
      - Conv2DTranspose
      - BatchNormalization
      - Activation
    blocks 4:
      - Conv2DTranspose
      - Activation
  paths 2:
    (same structure)
  paths 3:
    (same structure)
```

---

## 🧪 All Parameters

| Parameter          | Type          | Description                                               |
| ------------------ | ------------- | --------------------------------------------------------- |
| `filters`          | list of int   | Filters used in encoder (reversed for decoder)            |
| `kernel_size`      | tuple of ints | Must be odd numbers, e.g. (3,3), (5,5)                    |
| `batch_norm`       | bool          | Whether to apply BatchNormalization                       |
| `activation`       | str or None   | Hidden layer activation (e.g., 'relu', 'tanh', 'sigmoid') |
| `final_activation` | str or None   | Output activation for final layer in each decoder         |
| `out_channels`     | int           | Number of decoder branches (independent outputs)          |

---

## 📌 Notes

* The model follows the structure in **Figure 3(c)** from DeepCFD: encoder + multiple decoders.
* The final decoder layer skips batch norm (by design) for each output path.
* This layer class is flexible and can be extended into deeper or more task-specific models.

---

## ✅ License

MIT License

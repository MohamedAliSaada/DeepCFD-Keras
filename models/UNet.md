# U-Net with Skip Connections as Keras Custom Layer (double checked)

This repository provides a custom **UNet** implementation in Keras using functional encoder, decoder, and neck blocks. The architecture mimics the classical U-Net structure with skip connections, particularly useful for image segmentation, regression, or scientific modeling tasks (e.g., CFD, medical imaging).

---

## 🔧 Features

* Encoder and decoder blocks using sequential `Conv2D` and `Conv2DTranspose` layers
* Skip connections for rich feature fusion
* Customizable kernel size, filter depth, and activation functions
* Batch Normalization toggle for stable training
* Neck block supports separate control of bottleneck structure
* Layer inspection via `My_layers_is()` method

---

## 📦 Installation

No special installation required. Just ensure this file is accessible within your TensorFlow project directory.

---

## 🚀 Usage

### 1. Import the Layer

```python
from unet import UNet
```

### 2. Create a UNet instance

```python
model = UNet(
    filters=[16, 32, 64],         # encoder/decoder filters
    kernel_size=(3, 3),           # must be odd values
    batch_norm=True,              # toggle for BatchNormalization
    activation="relu",           # activation function for hidden layers
    final_activation="sigmoid",  # output layer activation
    out_channels=3                # number of output channels
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
Encoder Layers:
  - Conv2D
  - BatchNormalization
  - Activation
  - Conv2D
  - BatchNormalization
  - Activation
  - MaxPooling2D
  ...
Neck Layers:
  - Conv2D
  - BatchNormalization
  - Activation
  - Conv2D
  - BatchNormalization
  - Activation
Decoder Layers:
  Decoder Block 1:
    Upsample: Conv2DTranspose
    Concat: Concatenate
    - Conv2D
    - BatchNormalization
    - Activation
    - Conv2D
    - BatchNormalization
    - Activation
  Decoder Block 2:
    (same structure)
Final Output Layer:
  - Conv2D (1x1)
```

---

## 🧪 All Parameters

| Parameter          | Type          | Description                                               |
| ------------------ | ------------- | --------------------------------------------------------- |
| `filters`          | list of int   | Filters used in encoder (reversed for decoder)            |
| `kernel_size`      | tuple of ints | Must be odd numbers, e.g. (3,3), (5,5)                    |
| `batch_norm`       | bool          | Whether to apply BatchNormalization                       |
| `activation`       | str or None   | Hidden layer activation (e.g., 'relu', 'tanh', 'sigmoid') |
| `final_activation` | str or None   | Output activation for final layer                         |
| `out_channels`     | int           | Number of output channels (e.g., for multi-regression)    |

---

## 📌 Notes

* This U-Net version follows the structure in **Figure 3(b)** from DeepCFD.
* Pooling uses `MaxPooling2D`; upsampling uses `Conv2DTranspose`.
* Skip connections connect encoder outputs to corresponding decoder inputs.
* The model omits weight normalization and uses standard `BatchNormalization`.

---

## ✅ License

MIT License

"""
PVision AI: CNN Model Architecture Factory
Constructs transfer learning models with frozen backbones (Stage 1)
and supports unfreezing top layers for fine-tuning (Stage 2).

Supported Models:
- EfficientNetV2-S (Primary Model)
- ConvNeXt-Tiny (Prepared for future comparison)
- MobileNetV3-Large (Prepared for future comparison)
"""

from typing import Tuple, Optional
import os

def build_cnn_model(
    model_name: str = "efficientnetv2-s",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    dropout_rate: float = 0.3,
    pretrained: bool = True
):
    """
    Builds a CNN model with ImageNet pretrained backbone and custom classification head.
    The backbone is initially frozen for Stage 1 training.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError(
            "TensorFlow is required to build the CNN model. "
            "Please install TensorFlow in your VS Code environment: pip install tensorflow"
        )

    weights = "imagenet" if pretrained else None
    model_key = model_name.lower().replace("_", "-")

    if model_key in ["efficientnetv2-s", "efficientnetv2s"]:
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    elif model_key in ["convnext-tiny", "convnext"]:
        base_model = tf.keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    elif model_key in ["mobilenetv3-large", "mobilenetv3"]:
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}. "
            f"Supported options: ['efficientnetv2-s', 'convnext-tiny', 'mobilenetv3-large']"
        )

    # Freeze backbone for Stage 1 transfer learning
    base_model.trainable = False

    # Build Custom 7-Class Classification Head
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"PVision_{model_name.upper()}")
    return model, base_model


def configure_fine_tuning(
    model,
    base_model,
    unfreeze_layers: int = 50
):
    """
    Unfreezes the top N layers of the backbone for Stage 2 fine-tuning,
    keeping early low-level feature extraction layers frozen.
    """
    base_model.trainable = True

    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - unfreeze_layers)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Keep BatchNormalization layers frozen during fine-tuning for stability
            if isinstance(layer, type(model.layers[0])):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    return model, trainable_count

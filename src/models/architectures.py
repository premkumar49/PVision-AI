"""
PVision AI: Model Architecture Specifications
Blueprints for the Hybrid Framework:
1. PV Fault CNN: Deep Convolutional Neural Network for 7-class fault classification.
2. Solar Generation ANN: Multi-Layer Perceptron / Dense Regressor for AC power output forecasting.

NOTE: All model training will be performed manually by the user in VS Code.
      No training routines are executed automatically in this phase.
"""

from typing import Dict, Any

class PVFaultCNNBlueprint:
    """
    Architecture blueprint for Photovoltaic Fault Classification.
    Targeted for PyTorch / Torchvision (e.g., Custom CNN, ResNet-18, or MobileNetV3).
    """
    ARCHITECTURE_NAME = "PV_Fault_ResNet18"
    NUM_CLASSES = 7
    INPUT_CHANNELS = 3
    INPUT_SIZE = (224, 224)
    DROPOUT_RATE = 0.3

    RECOMMENDED_HYPERPARAMETERS = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "optimizer": "AdamW",
        "weight_decay": 1e-2,
        "epochs": 25,
        "loss_fn": "CrossEntropyLoss",
        "lr_scheduler": "CosineAnnealingLR"
    }


class SolarPowerANNBlueprint:
    """
    Architecture blueprint for Solar Generation Power Forecasting.
    Multi-Layer Perceptron (ANN) with residual or batch-normalized dense layers.
    """
    ARCHITECTURE_NAME = "Solar_Power_ANN"
    INPUT_DIM = 9  # Ambient Temp, Module Temp, Irradiation, Temp Diff, Interaction, Hour, Sin, Cos, Daytime
    HIDDEN_LAYERS = [128, 64, 32]
    OUTPUT_DIM = 1  # AC_POWER (kW)
    DROPOUT_RATE = 0.2
    ACTIVATION = "ReLU"

    RECOMMENDED_HYPERPARAMETERS = {
        "learning_rate": 5e-4,
        "batch_size": 64,
        "optimizer": "Adam",
        "loss_fn": "HuberLoss",  # Robust to solar spike outliers
        "epochs": 40
    }

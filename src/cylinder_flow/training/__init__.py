"""Training routines for encoders, pPINN, and POD."""
from cylinder_flow.training.encoder import (
    DeepSetsTrainer,
    DeepSetsTrainHistory,
    DeepSetsTrainingConfig,
    EncoderTrainingConfig,
    MLPTrainer,
    TrainHistory,
    set_training_seed,
    train_sensor_deepsets,
    train_sensor_mlp,
)
from cylinder_flow.training.pod import (
    PODConfig,
    PODTrainingConfig,
    train_pod_model,
    train_surface_pod,
)
from cylinder_flow.training.ppinn import (
    PPINNTrainHistory,
    PPINNTrainer,
    PPINNTrainingConfig,
    derive_ppinn_sampling_seeds,
    train_parametric_pinn,
    train_ppinn,
    validate_ppinn_splits,
)

__all__ = [
    "DeepSetsTrainer",
    "DeepSetsTrainHistory",
    "DeepSetsTrainingConfig",
    "EncoderTrainingConfig",
    "MLPTrainer",
    "TrainHistory",
    "set_training_seed",
    "train_sensor_deepsets",
    "train_sensor_mlp",
    "PODConfig",
    "PODTrainingConfig",
    "train_pod_model",
    "train_surface_pod",
    "PPINNTrainHistory",
    "PPINNTrainer",
    "PPINNTrainingConfig",
    "derive_ppinn_sampling_seeds",
    "train_parametric_pinn",
    "train_ppinn",
    "validate_ppinn_splits",
]

"""M20 quadruped fine-tuning config."""

import dataclasses
import pathlib

import flax.traverse_util
import numpy as np
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.quadruped_pi0_config as quadruped_pi0_config
import openpi.policies.quadruped_policy as quadruped_policy
import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class QuadrupedCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Loads Pi0 weights and initializes the new quadruped state adapter."""

    params_path: str

    @override
    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        result = {}
        for key, value in flat_loaded.items():
            if key in flat_ref:
                result[key] = value.astype(flat_ref[key].dtype) if value.dtype != flat_ref[key].dtype else value

        result.update({key: value for key, value in flat_ref.items() if key.startswith("state_adapter/")})

        return flax.traverse_util.unflatten_dict(result, sep="/")


def get_quadruped_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotQuadrupedDataConfig(DataConfigFactory):
        """Maps an M20 LeRobot dataset into the quadruped Pi0 policy format."""

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/front_image": "front_image",
                            "observation/bottom_image": "bottom_image",
                            "observation/state": "state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[quadruped_policy.QuadrupedInputs(model_type=model_config.model_type)],
                outputs=[quadruped_policy.QuadrupedOutputs()],
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
            )

    model = quadruped_pi0_config.QuadrupedPi0Config()
    return [
        TrainConfig(
            name="pi0_quadruped",
            model=model,
            data=LeRobotQuadrupedDataConfig(
                repo_id="your_hf_username/m20_quadruped_dataset",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=QuadrupedCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            freeze_filter=model.get_quadruped_freeze_filter(),
            num_train_steps=20_000,
            batch_size=32,
            fsdp_devices="auto",
        ),
        TrainConfig(
            name="pi0_quadruped_synthetic",
            model=quadruped_pi0_config.QuadrupedPi0Config(
                paligemma_variant="dummy",
                action_expert_variant="dummy",
            ),
            data=LeRobotQuadrupedDataConfig(
                repo_id="openpi/m20_quadruped_synthetic",
                base_config=DataConfig(prompt_from_task=True),
            ),
            num_train_steps=10,
            batch_size=2,
            num_workers=0,
            overwrite=True,
            exp_name="synthetic",
            wandb_enabled=False,
        ),
        TrainConfig(
            name="pi0_quadruped_synthetic_base",
            model=model,
            data=LeRobotQuadrupedDataConfig(
                repo_id="openpi/m20_quadruped_synthetic",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=QuadrupedCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            freeze_filter=model.get_quadruped_freeze_filter(),
            num_train_steps=1,
            batch_size=2,
            num_workers=0,
            fsdp_devices="auto",
            overwrite=True,
            exp_name="synthetic_base",
            wandb_enabled=False,
        ),
    ]

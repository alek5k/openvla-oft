"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import invert_gripper_actions

from xembench.data_collection.replay_buffer import ReplayBuffer
from xembench.datasets.dataset_utils import get_safe_action_chunk
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
import torchvision.transforms as tvt

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any], is_train: bool = True) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # debug_imgs = True
        # if debug_imgs:
        #     img.save("debug_img.png")  # Debugging line to save the image being processed
        #     from xembench.image_utils import depth_to_rgb_image, instance_segmentation_to_rgb_image
        #     depth_img = rlds_batch["observation"]["depth_primary"][0]
        #     depth_rgb = depth_to_rgb_image(depth_img)
        #     depth_rgb = Image.fromarray(depth_rgb)
        #     depth_rgb.save("debug_depth.png")  # Debugging line to save the depth image being processed
        #     egomask = rlds_batch["observation"]["egomask_primary"][0]
        #     egomask_rgb = instance_segmentation_to_rgb_image(egomask)
        #     egomask_rgb = Image.fromarray(egomask_rgb)
        #     egomask_rgb.save("debug_egomask_rgb.png")  # Debugging

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        # Loop over potentially multiple wrist images in the observation
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "image" in k and "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.is_train = train
        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False, # True
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                # depth_resize_size=resize_resolution,
                # egomask_resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch, self.is_train)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch), self.is_train)  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


def bounds_q99_normalize(x, q01, q99, eps=1e-8, clip=True, zero_unused=True):
    x = np.asarray(x, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32).reshape(1, -1)
    q99 = np.asarray(q99, dtype=np.float32).reshape(1, -1)

    y = 2.0 * (x - q01) / (q99 - q01 + eps) - 1.0
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if zero_unused:
        unused = np.isclose(q01, q99)
        if np.any(unused):
            x_norm = x_norm.copy()
            x_norm[:, unused.reshape(-1)] = 0.0
    return y



class ZarrDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        zarr_dataset_root: Path,
        zarr_dataset_name: str,
        resize_resolution: Tuple[int, int],
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        train: bool = True,
        image_aug: bool = False
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.zarr_dataset_root = zarr_dataset_root
        self.zarr_dataset_name = zarr_dataset_name
        self.zarr_dataset_path = zarr_dataset_root / zarr_dataset_name
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.train = train
        self.image_aug = image_aug
        self.src_buffer = ReplayBuffer.create_from_path(self.zarr_dataset_path, mode="r")
        
        action_data = self.src_buffer["action"] # shape is (N, 7)
        proprio_data_to_normalize = self.src_buffer["eef_pose_oxe"] # gripper is not normalized

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            self.zarr_dataset_name: {
                "action": {"q01": np.quantile(action_data, 0.01, axis=0), "q99": np.quantile(action_data, 0.99, axis=0)},
                "eef_pose_oxe": {"q01": np.quantile(proprio_data_to_normalize, 0.01, axis=0), "q99": np.quantile(proprio_data_to_normalize, 0.99, axis=0)}
            }
        }

        self.episode_ends = np.asarray(self.src_buffer.meta["episode_ends"], dtype=np.int64)

        H, W = resize_resolution

        self.train_image_augmentations = tvt.Compose([
            tvt.Resize((H, W)),  # match decode_and_resize
            tvt.RandomResizedCrop((H, W), scale=(0.9, 0.9), ratio=(1.0, 1.0)),  # fixed 90% area
            tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # tvt.RandomApply([tvt.GaussianBlur(kernel_size=5)], p=0.2),
        ])
        # RLDS aug uses a seed so all modalities are reproducible per sample. PyTorch aug randomness should be fine, but if we care about determinism across workers, 
        # we need to set a worker seed (worker_init_fn).
        # def seed_worker(worker_id):
        #     # base seed is set by DataLoader generator below
        #     worker_seed = torch.initial_seed() % 2**32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)
        # g = torch.Generator()
        # g.manual_seed(cfg.seed)  # your global seed
        # Then in dataloader:
        # worker_init_fn=seed_worker,
        # generator=g,
        # persistent_workers=True


        # I think during eval, these operations occur automatically, but leaving here for now.
        # see `prepare_images_for_vla` in `openvla_utils.py`
        # self.eval_tf = tvt.Compose([
        #     # tvt.Resize((H, W)),              # match decode_and_resize
        #     # CenterCropByArea(0.9),           # deterministic "center 90% area"
        #     # tvt.Resize((H, W)),              # resize back to model input size
        # ])


    def __len__(self):
        return self.src_buffer.n_steps


    def __getitem__(self, idx):
        # ALEKS NOTE:
        # Pytorch Dataset says that __getitems__ can also be used to get batches, speeding up data loading.
        # However, here we only implement single index access for simplicity.

        # One important note is that we need to make sure that the action chunk we get does not cross episode boundaries.
        # So we use the utility function `get_safe_action_chunk` to get the action chunk
        # Note that the RLDS dataset does not have this issue because it samples transitions within episodes (),
        # which also means that the dataset yields fewer samples (at the end of episodes).

        # action = self.src_buffer["action"][idx]
        action_chunk = get_safe_action_chunk(self.src_buffer, idx, NUM_ACTIONS_CHUNK, pad_zero_or_repeat="repeat")  # shape is (K, 7) make_dataset_from_rlds repeats the action.

        # TODO: ACTION NORMALIZATION (See make_dataset_from_rlds)
        # 'action_normalization_mask' = [True, True, True, True, True, True, False]
        # 'action_proprio_normalization_type' = NormalizationType.BOUNDS_Q99
        # TODO: implement normalize_action_and_proprio from data_utils.py

        action_chunk[:, :6] = bounds_q99_normalize(
            action_chunk[:, :6],
            q01=self.dataset_statistics[self.zarr_dataset_name]["action"]["q01"][:6],
            q99=self.dataset_statistics[self.zarr_dataset_name]["action"]["q99"][:6],
        )


        # TODO: Other trajectory level transforms?
        # 'absolute_action_mask' = [False, False, False, False, False, False, True]
        # does not seem to be used?
        # Go through `apply_trajectory_transforms` in more detail.
        

        # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
        gripper_actions = action_chunk[:, -1:] 
        action_chunk[:, -1:] =  invert_gripper_actions(np.clip(gripper_actions, 0, 1))

        image = Image.fromarray(self.src_buffer["agentview_rgb"][idx])
        if self.train and self.image_aug:
            image = self.train_image_augmentations(image)

        # debug_img = True
        # if debug_img:
        #     image.save("debug_img_zarr.png")

        # 
        instruction = self.src_buffer["language_instruction"][idx].lower()

        # assert np.array_equal(action, action_chunk[0]), "Current action does not match first action in action chunk!"
        # assert action_chunk.shape[0] == NUM_ACTIONS_CHUNK, "Action chunk length mismatch!"
        
        current_action = action_chunk[0]
        future_actions = action_chunk[1:]

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            # {"from": "gpt", "value": self.action_tokenizer(action)}, # SINGLE ACTION ONLY
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(len(action) + 1)] = IGNORE_INDEX # OLD - SINGLE ACTION ONLY
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=self.zarr_dataset_name, actions=action_chunk)

        # Add additional inputs
        if self.use_wrist_image:
            img_wrist = Image.fromarray(self.src_buffer["robot0_eye_in_hand_rgb"][idx])
            if self.train and self.image_aug:
                img_wrist = self.train_image_augmentations(img_wrist)
            pixel_values_wrist = self.image_transform(img_wrist)
            return_dict["pixel_values_wrist"] = pixel_values_wrist

        if self.use_proprio:

            robot_proprio = self.src_buffer["eef_pose_oxe"][idx]
            robot_proprio_norm = bounds_q99_normalize(
                robot_proprio,
                q01=self.dataset_statistics[self.zarr_dataset_name]["eef_pose_oxe"]["q01"],
                q99=self.dataset_statistics[self.zarr_dataset_name]["eef_pose_oxe"]["q99"],
            )
            gripper_qpos = self.src_buffer["robot0_gripper_qpos"][idx]
            proprio = np.concatenate([robot_proprio_norm, gripper_qpos], axis=0)
            return_dict["proprio"] = proprio

        return return_dict

class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

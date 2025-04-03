from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class SsLifteV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front view RGB camera observation.',
                        ),
                        'left_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Left view RGB camera observation.',
                        ),
                        'right_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Right view RGB camera observation.',
                        ),
                        'top_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Top view RGB camera observation.',
                        ),
                        'eef_pos': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='End-effector position in 3D space (x, y, z).',
                        ),
                        'eef_quat': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float32,
                            doc='End-effector orientation as quaternion (w, x, y, z).',
                        ),
                        'gripper_pos': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='Gripper finger positions (2 values representing gripper state).',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint positions for the 7-DOF Franka arm.',
                        ),
                        'blocks_poses': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Poses of blocks in the environment (position and orientation).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x end-effector velocity/pose delta, '
                            '1x gripper open/close command].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction describing the task to perform.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),  # No path parameter needed as there are no splits
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data
            data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([step['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['image'],
                        'left_image': step['left_image'],
                        'right_image': step['right_image'],
                        'top_image': step['top_image'],
                        'eef_pos': step['eef_pos'].astype(np.float32),
                        'eef_quat': step['eef_quat'].astype(np.float32),
                        'gripper_pos': step['gripper_pos'].astype(np.float32),
                        'joint_pos': step['joint_pos'].astype(np.float32),
                        'blocks_poses': step['blocks_poses'].astype(np.float32),
                    },
                    'action': step['action'],
                    'discount': step.get('discount', 1.0),  # Use provided discount or default to 1.0
                    'reward': step.get('reward', float(i == (len(data) - 1))),  # Use provided reward or default
                    'is_first': step.get('is_first', i == 0),  # Use provided flag or default
                    'is_last': step.get('is_last', i == (len(data) - 1)),
                    'is_terminal': step.get('is_terminal', i == (len(data) - 1)),
                    'language_instruction': step['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return episode_path, sample

        # Find all episodes in the demonstrations directory
        # Assuming your demonstrations are saved in a structure like demonstrations/env_name/timestamp/episode_*.npy
        # Adjust the pattern if your directory structure is different
        episode_paths = glob.glob('demonstrations/**/**/episode_*.npy', recursive=True)
        
        if not episode_paths:
            raise ValueError("No episode files found. Please check the path to your demonstrations.")
        
        print(f"Found {len(episode_paths)} episodes")

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
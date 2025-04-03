# Franka Lift Environment Dataset

## Overview
This dataset contains demonstrations of a Franka robot performing lifting tasks in a simulated environment. The demonstrations were collected using a PlayStation 4 controller for teleoperation. The dataset is designed for training reinforcement learning and imitation learning agents to perform manipulation tasks.

## Environment
- **Robot**: 7-DOF Franka Emika Panda arm with gripper
- **Task**: Lifting objects in a table-top environment
- **Simulator**: [Your simulator name, e.g., MuJoCo/PyBullet/Isaac Gym]
- **Interface**: Manual demonstrations collected via PlayStation 4 controller

## Dataset Statistics
- **Number of demonstrations**: [Your number, e.g., 100]
- **Total steps**: [Your number, e.g., 15,000]
- **Average episode length**: [Your number, e.g., 150 steps]
- **Success rate**: [Your number, e.g., 95%]

## Observation Space
The observation space consists of:
- **image**: Front camera RGB image (224×224×3)
- **left_image**: Left camera RGB image (224×224×3)
- **right_image**: Right camera RGB image (224×224×3)
- **top_image**: Top camera RGB image (224×224×3)
- **eef_pos**: End-effector 3D position (x, y, z)
- **eef_quat**: End-effector orientation as quaternion (w, x, y, z)
- **gripper_pos**: Gripper finger positions (2 values)
- **joint_pos**: 7 joint positions of the Franka arm
- **blocks_poses**: Poses of objects in the environment (14 values)

## Action Space
The action space is 7-dimensional:
- **Dimensions 0-5**: End-effector velocity/pose delta
- **Dimension 6**: Gripper command (open/close)

## Example Trajectories

![Example Trajectory 1](example_images/trajectory1.png)
*Caption: A successful trajectory of lifting a block*

![Example Trajectory 2](example_images/trajectory2.png)
*Caption: Another successful trajectory showing different viewpoints*

## Data Collection Process
The data was collected by human operators using a PlayStation 4 controller to teleoperate the Franka robot in simulation. The demonstrations focus on successfully grasping and lifting objects from the tabletop. Each demonstration includes multi-view RGB observations and corresponding robot state information.

## Usage
This dataset follows the RLDS (Reinforcement Learning Datasets) format for compatibility with standard reinforcement learning benchmarks. It can be used for:
- Imitation Learning
- Reinforcement Learning
- Behavior Cloning
- Inverse Reinforcement Learning

## License
[Your License, e.g., MIT, CC BY 4.0, etc.]
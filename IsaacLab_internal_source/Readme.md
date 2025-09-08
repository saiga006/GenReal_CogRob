# Franka Kitchen Pick and Place Task implemented in IsaacLab using Behavior Cloning based Imitation Learning Policy

Overview
--------
- This repository contains the implementation, data-generation pipelines, and trained behavior-cloning models for a kitchen pick-and-place task using IsaacLab.
- Task: a Franka Emika manipulator picks a tomato soup can (container) from inside a fridge and places it inside the microwave cavity.
- We reused the IsaacLab sample Franka Lift Cube task and adapted to our kitchen task by modifying the internal source code of IsaacLab.
- To reuse the code, please clone [IsaacLab 2.1.1 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.1.1) and use IsaacSim 4.5 version and replace existing source code with the files and folders given under IsaacLab_internal_source.
- The other folders outside the IsaacLab_internal_source are meant to create the similar kitchen scene with kinova gen3 7DOF arm as an external project in IsaacLab, primarily meant for RL PPO implementation(not implemented currently).

Highlights
----------
- A custom kitchen scene (shelf, fridge, microwave, tomato soup can) and Franka spawn is provided.
- Teleoperated demonstrations were recorded, annotated (Isaac Mimic tooling), and expanded with isaacmimic gen to produce ~300 annotated demonstrations (from 10 teleoperated trials, parallelized generation).
- Behavior cloning training was performed using robomimic with multiple configurations (state-based and visuomotor image-based policies). Several 'fast' and 'ultrafast' variants were used to trade off training time vs accuracy.

Primary files & commands
------------------------

1. Record teleop demos (example using teleoperation script):

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Lift-Kitchen-Franka-IK-Rel-v0 \
  --num_envs 1 --enable_cameras
```

2. Annotate teleop demos (example):

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Kitchen-Lift-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --input_file ./datasets/kitchen_task_vision_11_.hdf5 \
  --output_file ./datasets/annotated_dataset_modified2.hdf5 \
  --enable_cameras
```

3. Generate expanded dataset (isaacmimic gen, 10 parallel envs -> 300 trials):

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --enable_cameras --headless --num_envs 10 --generation_num_trials 300 \
  --input_file ./datasets/annotated_dataset_modified2.hdf5 \
  --output_file ./datasets/generated_dataset_large.hdf5
```

4. Data split use the script from robomimic repo (train/validation, 1:10 ratio):

```bash
./isaaclab.sh -p split_train_val.py \
  --dataset ./datasets/generated_dataset_sample_300.hdf5 --ratio 0.1
```

5. Example training (state-based BC):

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Lift-Kitchen-Franka-IK-Rel-v0 \
  --algo bc \
  --dataset ./datasets/generated_dataset_sample_300.hdf5
```

6. Example play / evaluation (state-based BC):

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Lift-Kitchen-Franka-IK-Rel-v0 --num_rollouts 50 \
  --checkpoint /path/to/models/model_epoch_best_validation.pth \
  --horizon 2500 --enable_cameras
```

7. Visuomotor ultra-fast training/play example:

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Lift-Kitchen-Franka-IK-Rel-Visuomotor-UltraFast-v0 \
  --algo bc --dataset ./datasets/generated_dataset_sample_300.hdf5

./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Lift-Kitchen-Franka-IK-Rel-Visuomotor-UltraFast-v0 --num_rollouts 50 \
  --checkpoint /path/to/models/best_validation_epoch.pth --enable_cameras
```

## Development notes

- A mimic env cfg and mimic env were added to support dataset expansion (isaacmimic gen) from a small set of annotated trials. The mimic env supports generating annotated datasets with cameras enabled and parallel envs.
- Subtask breakdown in mimic environment: move to fridge, grasp tomato can, move to microwave, place in microwave (see `franka_kitchen_lift_visuomotor_mimic_env_cfg.py`).
- Custom success detection function `object_in_microwave_and_hand_out()` verifies tomato can placement, gripper openness, and hand clearance (see `kitchen_joint_pos_env_cfg.py`).
- Custom observation functions: `object_position_in_robot_root_frame()`, `ee_frame_pos()`, `ee_frame_quat()`, `gripper_pos()` (see `mdp/observations.py`).
- Asset reuse from Lightwheel: Franka kitchen scene reuses assets from Kinova folder for consistency and compatibility (see `config/kinova/assets/`).
- Scripts & logs used to generate the 300 annotated demos from 10 teleoperated trials were added; logs are available in `log_dir/`.
- Robomimic BC configs were added for state-based and visuomotor policies (multiple RNN and ResNet-18 + R3M variants).
- Lighting randomization, scene scaling for parallel envs, and wrist-camera optimizations were added for better generalization and stable data collection.

## Folder layout

- IsaacLab_internal_source/
  - lift/ — core lift task code and configs
    - lift/lift_env_cfg.py — base environment config with scene setup, MDP settings
    - lift/mdp/ — custom MDP components
      - custom_events.py — light intensity randomization for scene variation
      - observations.py — custom observation functions (object position, EE frame, gripper pos)
      - rewards.py, terminations.py — reward and termination functions
    - lift/config/franka/ — Franka-specific envs and kitchen scene
      - kitchen_scene.py — kitchen scene (table, shelf, fridge, microwave, tomato soup can) with Franka spawn
      - kitchen_joint_pos_env_cfg.py — joint-position kitchen env for Franka with camera config
      - kitchen_ik_rel_env_cfg.py — relative IK env for Franka (teleop/eval)
      - kitchen_teleop_env_cfg.py — teleoperation env + tweaks for demo collection
      - agents/ — agent config files for training (robomimic / rsl / sb3 / skrl)
      - __init__.py — Gym environment registrations
    - lift/config/kinova/ — original kinova assets reused by the kitchen scene
  - isaaclab_mimic/ — mimic environment for dataset generation
    - envs/ — mimic environment implementations
      - franka_kitchen_lift_visuomotor_mimic_env_cfg.py — visuomotor mimic env config with subtask breakdown
      - franka_kitchen_lift_visuomotor_mimic_env.py — environment wrapper for IK-relative control
      - __init__.py — Gym registration for mimic environments
  - imitation_learning/isaaclab_mimic/ — mimic annotation + generation scripts
    - annotate_demos.py, generate_dataset.py, consolidated_demo.py — utilities for annotation and dataset generation
  - trained_bc_models/ — trained BC artifacts and zipped bundles; contains configs, logs, models and videos
  - log_dir/ — logs produced during dataset generation and training pipelines

## Video Output of the BC (state & visuomotor) rollouts:
[Please find the model results here](https://app.box.com/s/dp3khsxa8kkebke9anjs2aj4w1eoqn5v)

## Behavior Cloning: Key Configurations & Parameters

This section summarizes the most important configuration options for behavior cloning (BC) in IsaacLab/robomimic tasks:

- **algo_name**: Algorithm type (usually `bc` for behavior cloning).
- **experiment**: Controls logging, saving, validation, and rollout settings.
  - `validate`: Enables validation during training.
  - `logging`: Options for TensorBoard and terminal output.
  - `save`: When and how to save checkpoints (e.g., every N epochs, on best validation).
  - `epoch_every_n_steps`: Number of steps per training epoch.
- **train**: Data loading and training loop settings.
  - `num_data_workers`: Number of CPU workers for data loading (higher = faster, up to available cores).
  - `hdf5_cache_mode`: What to cache in RAM (`all`, `low_dim`, or `None`).
  - `batch_size`: Number of samples per training batch (higher = better GPU utilization, but more VRAM needed).
  - `num_epochs`: Total number of training epochs.
  - `seq_length`: Length of input sequence for RNNs (higher = more temporal context).
- **algo**: Model architecture and optimization.
  - `optim_params`: Learning rate, decay schedule, and regularization.
  - `actor_layer_dims`: Hidden layer sizes for the policy network.
  - `gmm`: Number of Gaussian Mixture Model modes (higher = more action diversity).
  - `rnn`: RNN settings (enabled, hidden size, layers, etc.).
- **observation**: Defines input modalities and encoders.
  - `modalities`: Which observation types are used (`low_dim`, `rgb`, etc.).
  - `encoder`: Backbone network (e.g., `ResNet18Conv`, `R3MConv`), feature dimension, pooling, and randomization/cropping.

### Impact of Key Parameters
- **batch_size**: Larger batches speed up training and improve stability, but require more GPU memory.
- **num_data_workers**: More workers reduce data loading bottlenecks, especially for image-based BC.
- **hdf5_cache_mode**: Caching images (`all`) is fastest but uses a lot of RAM; `low_dim` is safer for large datasets.
- **gmm.num_modes**: More modes allow the policy to represent more complex/multimodal actions.
- **rnn.hidden_dim/layers**: Larger RNNs capture more temporal dependencies but use more memory.
- **encoder.backbone_class**: Choice of backbone affects visual feature quality and training speed.

See the config files in `isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/robomimic/` for examples and recommended settings for different hardware and task complexity.

## Notes

- Camera config: wrist camera must have valid `width` and `height` values — defaults used in this project are `88x88`.
- Asset paths: Franka kitchen reuses assets in `lift/config/kinova/assets/`; keep the relative paths intact.
- Mimic generation: generating datasets with images can be memory/GPU intensive. Use headless mode and set `num_envs` according to available GPU resources.
- Validation: BC checkpoints were selected by validation loss; ensure the model and config match the observation/action spaces when loading.

## Reproducibility checklist

1. Ensure Isaac Sim / IsaacLab environment is installed and `isaaclab.sh` works.
2. Record teleop demos using `scripts/environments/teleoperation/teleop_se3_agent.py`.
3. Annotate demos via `annotate_demos.py`.
4. Run `generate_dataset.py` with `--num_envs` parallelization to expand the annotated trials.
5. Split dataset using the robomimic split script.
6. Run robomimic training scripts (configs available in `agents/` and `trained_bc_models/*/config.json`).
7. Evaluate with `robomimic/play.py` using the selected checkpoint.


## Dataset

[Please find the dataset here](https://app.box.com/s/zg53efxfykoptozaf4a7fax8njdi090a)

The dataset consists of annotated demonstrations for the kitchen pick-and-place task, generated using teleoperation and expanded via the mimic environment. Key files:

- `datasets/kitchen_task_vision_11_.hdf5` — raw teleop demonstrations
- `datasets/annotated_dataset_modified2.hdf5` — annotated dataset after processing
- `datasets/generated_dataset_large.hdf5` — expanded dataset (300 trials, 10 parallel envs)

Each dataset contains:
- Low-dimensional observations (eef pose, joint positions, object pose, actions)
- Camera images (if enabled)
- Subtask signals for task segmentation (move to fridge, grasp, move to microwave, place it inside microwave) used by isaac mimic to generate new datasets from 10 trials.

See the scripts in `imitation_learning/isaaclab_mimic/` for annotation and generation details.

## References

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): Core simulation and RL environment
- [Robomimic](https://github.com/ARISE-Initiative/robomimic): Behavior cloning and imitation learning framework
- [Isaac Mimic](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/teleop_imitation.html#training-an-agent): Annotation, dataset expansion, and imitation learning documentation
- [Franka Emika Panda](https://www.franka.de/): Robot platform used in the task
- [Kinova IsaacLab Sim2Real](https://github.com/louislelay/kinova_isaaclab_sim2real/tree/main): Sim2real pipeline for Kinova and IsaacLab
- [Robomimic Pretrained Representations](https://robomimic.github.io/docs/v0.4/tutorials/pretrained_representations.html): R3M model usage in BC training
- [R3M Model](https://sites.google.com/view/robot-r3m): Pretrained R3M representation for robot learning
- [IsaacLab Tutorials](https://lycheeai-hub.com/isaac-lab): IsaacLab tutorials and guides
- [IsaacLab Task Workflows](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/overview/core-concepts/task_workflows.html): IsaacLab task workflow documentation
- [IsaacLab Official Tutorials](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/tutorials/index.html): IsaacLab official tutorials
- Lightwheel, "Lightwheel Kitchen: 3D Kitchen Asset Collection for NVIDIA Isaac Sim," Version v1, 2025. [Online]. Available: https://github.com/LightwheelAI/Lightwheel_Kitchen — Kitchen assets used in this project

For further details, see the documentation and code comments in the respective modules.

## Contributors

This project is maintained and developed as part of Cognitive Robotics Course Project.

Below are the contributions:

[Sai Mukkundan Ramamoorthy](mailto:saimukkundan.ramamoorthy@gmail.com) - Kitchen scene setup script in IsaacLab, Parallel environment Simulation setup with franka, along with Behavior Cloning training and validation with state based and visuomotor based policy. 

[Aaron Cuthinho](mailto:aaron.cuthinho@smail.inf.h-brs.de) - for teleoperation, dataset annotation, dataset augmentation creation and RL PPO scripts.

[Saloni Pathak](mailto:pathak.saloni.de@gmail.com) - Kitchen Scene setup script with kinova arm in Robocasa, teleoperation script setup in robocasa.

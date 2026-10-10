# Figure 2 arrow-free components

This directory contains 15 independently cropped, 300 DPI, transparent PNG
assets extracted from the visual language of Figure 2. External titles and
scene captions are omitted so labels can be added during manual composition.
Mathematical output annotations are retained. Flow and control arrows are
intentionally omitted; lines inside the neural networks are structural
connections rather than flow arrows.

| File | Component |
| --- | --- |
| `01_lidar_sequence.png` | Multi-frame LiDAR sequence |
| `02_robot_state.png` | Goal, velocity, and motion-mode state |
| `03_lidar_encoder.png` | LiDAR MLP--GRU encoder |
| `04_state_encoder.png` | State MLP encoder |
| `05_fused_embedding.png` | Fused embedding bar |
| `06_td3_actor.png` | TD3 Actor network |
| `07_td3_critic.png` | TD3 Critic network |
| `08_neural_pvo.png` | Neural-PVO risk head |
| `09_safety_filter.png` | Risk-constrained safety filter |
| `10_4wis_mapper.png` | 4WIS action/mode mapper |
| `11_safe_velocity.png` | Safe body-velocity command |
| `12_geometry_proxy.png` | Current geometric risk proxy |
| `13_irsim_training.png` | IRSim training scene |
| `14_reward_transition.png` | Reward and transition block |
| `15_gazebo_validation.png` | Gazebo validation scene |

The editable source is `figure2_components.tex`. Rebuild and export with:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error figure2_components.tex
pdftocairo -png -transp -r 300 figure2_components.pdf component
```

Some image viewers display transparent pixels as black. Black labels become
visible normally after placing the PNG on a white or light-colored canvas.

## Connected reference

`figure2_connected_reference.pdf` is an optional connected three-lane draft
assembled from the PNG assets. It is not used by the current paper. Its editable TikZ source is
`figure2_connected_reference.tex`. Gray arrows denote forward feature/action
flow, red arrows denote risk inference, blue arrows denote training feedback,
and the dashed green arrow denotes high-fidelity Gazebo deployment validation.

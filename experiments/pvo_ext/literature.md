Brief literature summary (high-level directions)

Relevant directions for enhancing PVO (Predictive/Probabilistic Velocity/Collision models):

1) Uncertainty Estimation and Risk-Aware Decision Making
- Use Bayesian approximations (MC Dropout, deep ensembles) to quantify prediction uncertainty.
- Allows safety-aware planning: avoid actions with high predicted collision probability and high uncertainty.

2) Multi-horizon / Trajectory-conditioned Predictions
- Predict collision probabilities for multiple candidate trajectories or a short action sequence instead of a single-step action.
- Improves planning since downstream controller can select low-risk trajectories.

3) Attention & Spatial-Temporal Fusion
- Use attention across lidar angular sectors and time to focus on salient moving obstacles.
- Transformer/Temporal Conv alternatives often improve long-range dependencies.

4) Contrastive/Self-supervised Pretraining
- Pretrain encoder on large unlabeled lidar sequences to learn robust features (contrastive losses), then fine-tune for collision prediction.

5) Integrating PVO with a Safety Layer (Shield)
- Use PVO predictions to filter or modify RL-chosen actions (action correction / safety filter) rather than only as an input feature.

Proposed quick experiment for this repo (feasible without large compute):
- Implement a Monte-Carlo Dropout variant of the existing `PVONetwork` to estimate mean collision prob and predictive uncertainty.
- Train both the baseline PVONetwork (as-is) and Uncertainty-PVO (short epochs) on the existing `pvo_dataset.npz`.
- Run short evaluation episodes in the existing simulator for three setups:
  A) No-PVO (zero features) — baseline without learned predictor
  B) PVO (existing PVONetwork loaded) — original approach
  C) PVO-UQ (our MC-Dropout extension) — use mean prediction and uncertainty to pick safer actions

Metrics:
- Success rate (fraction of episodes that reach goal)
- Collision rate (fraction of episodes that end by collision)

Notes & assumptions:
- This is a focused, small-scale verification to show feasibility and relative differences. For publication-worthy results, increase training time, tune hyperparameters, and run more seeds.

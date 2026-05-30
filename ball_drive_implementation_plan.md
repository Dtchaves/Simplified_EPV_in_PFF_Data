Below is a strict implementation plan to match the original EPV paper as closely as possible, assuming your pass modules already provide normalization, pitch control/influence, dynamic pressure lines, reward labeling, splits, and model-training utilities.

## Target: implement exactly three ball-drive components

You need:

[
P(O_\delta = 1 \mid A=\delta,T_t)
]

[
E[G \mid A=\delta,O_\delta=1,T_t]
]

[
E[G \mid A=\delta,O_\delta=0,T_t]
]

Then combine them as:

[
E[G \mid A=\delta,T_t]
======================

p_\delta V_{\delta,\text{success}}
+
(1-p_\delta)V_{\delta,\text{failed}}
]

where (\delta) is ball drive, (O_\delta=1) means the team keeps control, and (O_\delta=0) means loss of ball control. The paper explicitly defines ball-drive outcome as successful ball drive vs. loss of control and combines the components into the decomposed EPV framework.

---

# 1. Reuse the exact existing infrastructure

Because your pass modules are already implemented, **do not create a new parallel preprocessing system**. Reuse:

* match-level train/validation/test split;
* coordinate normalization so the acting team attacks left-to-right;
* tracking snapshot extraction at action time;
* pitch control at ball location;
* pitch influence at ball location;
* dynamic pressure line extraction;
* long-term reward labeling;
* open-play filtering;
* model-training loop and calibration/evaluation utilities.

The original paper uses match-level splits, then shuffles events only inside the training set to reduce bias from nearby events in time. It uses validation for model selection and the test set as hold-out.

---

# 2. Build the canonical ball-drive sample table

The original uses human-labeled event data containing pass, ball drive, and shot events, aligned with 10 Hz optical tracking data. A ball drive is successful if the team does not lose control after the action occurs.

Create one canonical table:

```text
ball_drive_samples
- match_id
- period
- action_id / event_id
- original_event_id
- segment_id
- team_id
- player_id
- start_frame_id
- start_time
- end_time
- duration
- start_ball_x
- start_ball_y
- y_success
- reward_G
- split
```

Important: if your data has `carry_outcome`, use it as the primary success label. Falling back to “next event team equals same team” is less faithful and should only be used if the provider outcome is missing or demonstrably unreliable.

---

# 3. Split long ball drives into 1-second actions

The paper says that ball drives lasting more than 1 second are split into individual ball drives of 1-second duration.

Implementation rule:

```text
if duration <= 1s:
    keep one sample at action start
else:
    create non-overlapping 1s samples:
        [t0, t0+1]
        [t0+1, t0+2]
        ...
```

For each segment, use the **tracking snapshot at the segment start** as (T_t).

For success:

```text
y_success = 1 if the ball-carrier's team still controls the ball after the segment/action
y_success = 0 if control is lost after the segment/action
```

Avoid overlapping windows, frame-by-frame carry samples, or inferred synthetic carries from tracking alone. That would no longer match the original setup.

One ambiguity: the paper does not clearly state whether sub-1-second remainders after splitting long carries are kept or discarded. The most literal implementation is to keep original drives of at most 1 second and split longer drives into full non-overlapping 1-second chunks.

---

# 4. Apply the original open-play filter

Train only on open-play passes, ball drives, and shots. The original defines set-piece actions as actions observed within 5 seconds of the start of a direct/indirect free kick, corner, throw-in, or penalty. Those are ignored for model training. However, goals from set pieces are still used when assigning rewards to previous open-play actions.

So implement:

```text
is_set_piece_action =
    action_time <= set_piece_start_time + 5s
    for direct free kick / indirect free kick / corner / throw-in / penalty

training_sample = ball_drive and not is_set_piece_action
```

For reward labeling, still scan all goals, including set-piece goals.

---

# 5. Assign the long-term reward (G)

Use the same reward definition as your pass EPV modules.

The paper assigns:

```text
G = +1 if the action team scores the next goal within 15s
G = -1 if the opponent scores the next goal within 15s
G =  0 if no goal is observed within 15s or the half ends first
```

The original possession definition starts from kick-off and ends when a goal is observed or the half ends, but then adds a 15-second reset horizon: actions more than 15 seconds before the next goal receive reward 0.

Use exactly the same label for both successful and failed ball-drive expectation models.

---

# 6. Build the exact scalar feature set

Unlike pass modules, ball-drive models do **not** use SoccerMap surfaces. They use scalar spatial/contextual features and shallow neural networks. The paper says ball-drive and shot components use shallow neural networks over spatial/contextual features to produce single-valued predictions.

For the ball-drive probability model, use only the original feature family:

```text
ball_x
ball_y
angle_between_ball_and_opponent_goal
distance_between_ball_and_opponent_goal
pitch_control_attacking_team_at_ball
pitch_influence_defending_team_at_ball
closest_attacking_team_vertical_pressure_line_to_ball
closest_defending_team_vertical_pressure_line_to_ball
```

The appendix marks ball location, ball-goal angle, ball-goal distance, attacking-team pitch control at the ball, defending-team pitch influence at the ball, and closest attacking/defending pressure lines at the ball as features for ball-drive probability / expectation.

For the ball-drive expectation models, use the same feature vector plus:

```text
p_ball_drive_success
```

The paper explicitly says the ball-drive expectation model uses the same input dataset and feature extractor as the probability model, with the addition of the ball-drive probability estimate.

Do **not** include `team_phase`, `position_name`, `preferred_foot`, `height`, `weight`, `carry_type`, player ID, team ID, or player/team priors if you want the original implementation. Those would be reasonable extensions, but not original-paper faithful.

---

# 7. Train the ball-drive probability model

Model:

```text
input: DP feature vector

Dense layer
ReLU
Dense layer
ReLU
Dense layer with 1 output
Sigmoid

output: p_success
loss: binary cross-entropy / log loss
```

The original describes the ball-drive probability model as two fully connected layers, each followed by ReLU, with a single-neuron sigmoid output.

Training setup:

```text
optimizer: Adam
beta1 = 0.9
beta2 = 0.999
learning_rate grid = {1e-3, 1e-4, 1e-5, 1e-6}
batch_size grid = {16, 32}
early_stopping_delta = 1e-3
model selection: validation loss
final report: test set
```

These are the original model-selection settings.

Expected sanity check: the original dataset had 413,123 ball-drive samples and 90.60% success. Your rate does not need to match, but if it is far from this, inspect action mapping and success labeling.

---

# 8. Train two separate ball-drive expectation models

Use the same pattern as pass succeeded/failed EPV.

The paper says pass value is learned with separate models for successful and unsuccessful actions, and the ball-drive expectation follows the analogous approach.

So train:

```text
drive_value_success_model:
    train only on ball-drive samples with y_success = 1
    target = reward_G

drive_value_failed_model:
    train only on ball-drive samples with y_success = 0
    target = reward_G
```

Architecture:

```text
input: DP features + p_ball_drive_success

Dense layer
ReLU
Dense layer
ReLU
Dense layer with 1 output
Sigmoid
linear transform: value = 2 * sigmoid_output - 1

loss: MSE(value, reward_G)
```

The paper applies sigmoid to get ([0,1]), then linearly transforms to ([-1,1]), and optimizes MSE against the observed reward.

Do not force the successful model to be positive or the failed model to be negative. The paper explicitly avoids assuming that successful actions must have positive reward or failed actions must have negative reward in the analogous pass-value setup.

---

# 9. Combined inference API

Your final ball-drive module should expose:

```python
def predict_ball_drive_epv(state):
    x_dp = build_ball_drive_features(state)

    p_success = drive_probability_model(x_dp)

    x_de = concat(x_dp, p_success)

    v_success = drive_value_success_model(x_de)
    v_failed = drive_value_failed_model(x_de)

    drive_epv = (
        p_success * v_success
        + (1.0 - p_success) * v_failed
    )

    return {
        "p_drive_success": p_success,
        "v_drive_success": v_success,
        "v_drive_failed": v_failed,
        "drive_epv": drive_epv,
    }
```

Then your full EPV model consumes:

```text
P(A = drive | T_t) * E[G | A = drive, T_t]
```

where `P(A = drive | T_t)` comes from your action-selection model, not from the ball-drive success probability model.

---

# 10. Evaluation checklist

For `drive_probability_model`:

```text
cross-entropy / log loss
ECE
calibration curve with 10 equal-sized bins
success rate by prediction bin
```

For the value models:

```text
MSE
ECE-style calibration of predicted reward vs. observed reward
separate plots for success and failed ball drives
combined drive EPV calibration
```

The original reports calibration plots for pass/ball-drive probability, ball-drive successful/missed EPV, pass+ball-drive joint estimation, and final EPV.

Also, do not automatically apply post-hoc calibration to ball-drive probability. The original only applied post-training temperature scaling to pass success probability and pass selection probability.

---

# 11. Suggested implementation order

1. **Freeze the shared preprocessing contracts**: same match split, same coordinate normalization, same reward labels, same set-piece filtering.
2. **Create `BallDriveSegmentBuilder`**: select carry/drive events and split long drives into 1-second samples.
3. **Create `BallDriveLabelBuilder`**: assign `y_success` and `reward_G`.
4. **Create `BallDriveFeatureBuilder`**: return exactly the DP feature vector.
5. **Train DP**: binary success probability.
6. **Generate frozen DP predictions** for train/validation/test.
7. **Create DE datasets**: append `p_ball_drive_success`; split into success and failed subsets.
8. **Train DE-success and DE-failed** with MSE on (G).
9. **Create combined inference wrapper**.
10. **Run calibration diagnostics**.
11. **Plug into action-selection-weighted full EPV**.

The main “don’t accidentally approximate the paper” rules are: use event-defined ball drives, split long drives into non-overlapping one-second actions, use only original scalar features, train separate success/failed value models, keep the 15-second reward horizon, and exclude set-piece actions from training while still using all goals for reward labeling.

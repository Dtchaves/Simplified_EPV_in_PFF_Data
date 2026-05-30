Great — since your pass and ball-drive components are already done, the remaining pieces are:

1. **shot value / shot EPV module**, i.e. (E[G \mid A=\varsigma, T_t]);
2. **action selection probability**, i.e. (P(A=a \mid T_t)) for (a \in {\text{pass}, \text{ball drive}, \text{shot}}).

In the Fernández EPV framework, the final EPV is obtained by combining the value of pass, ball-drive, and shot actions, weighted by the probability that each action is selected in the current state. The paper explicitly decomposes EPV into these action components and estimates them separately, then merges them back into one calibrated EPV estimate.

---

# 1. Shot module: what you should implement

The shot module should estimate:

[
V_{\text{shot}}(T_t)=E[G \mid A=\varsigma, T_t]
]

where (A=\varsigma) means “a shot is attempted”, and (G \in {-1,0,1}) is the long-term possession reward.

A key point: **the paper calls this an expected goals model, but in the full EPV framework it is not just a classic immediate xG model**. The final “Shot EPV” model is trained with the same long-term reward definition used for pass and ball-drive EPV: +1 if the acting team scores next, −1 if the opponent scores next, and 0 if no goal is observed within the possession/reset horizon. The paper’s baseline event-data xG model is used as an **input feature**, not as the final shot EPV model.

## 1.1 Build the shot training dataset

For every observed **open-play shot event**, create one sample.

Each sample should contain:

```text
match_id
event_id
frame_id
team_id
player_id
normalized tracking snapshot at shot time
shot event metadata
long-term reward target G
```

The paper uses tracking and event data from **633 EPL matches**, with tracking at **10 Hz**, and focuses on pass, ball-drive, and shot events. It reports 13,735 shot events split into 8,240 training, 2,800 validation, and 2,695 test examples.

### Filtering

To reproduce the paper, use only **open-play actions** for the component models. The paper ignores actions occurring during set pieces and defines a set-piece action as one observed **5 seconds or less** after the start of a direct/indirect free kick, corner, throw-in, or penalty. Goals from set pieces are still used for reward labeling, but set-piece actions themselves are excluded from training.

So for your shot module:

```python
is_set_piece_action = (
    event_time <= set_piece_start_time + 5 seconds
    and set_piece_type in {free_kick, corner, throw_in, penalty}
)

keep_sample = event_type == "shot" and not is_set_piece_action
```

## 1.2 Normalize the game state

For every shot, normalize coordinates so the team taking the action attacks **left to right**. This is important because all spatial features in the paper assume that the attacking team is moving toward the right-side goal. The appendix states that all features are normalized under this left-to-right attacking convention.

For each shot frame:

```text
attacking team = team_id of shooter
defending team = opponent
opponent goal = right-side goal after normalization
own goal = left-side goal after normalization
ball location = shot origin / current ball location
```

## 1.3 Create the long-term shot reward target

For the final shot EPV module, do **not** train only on `shot_outcome == goal`.

Instead, assign the same long-term reward used for EPV components:

```text
G = +1 if the shooter’s team scores the next goal within epsilon seconds
G = -1 if the opponent scores the next goal within epsilon seconds
G = 0 otherwise
```

The paper uses:

```text
epsilon = 15 seconds
```

because 15 seconds corresponds to the average duration of standard soccer possessions in the available matches. Actions more than 15 seconds before the next goal receive reward 0.

Implementation sketch:

```python
def assign_long_term_reward(action, next_goal, epsilon=15.0):
    if next_goal is None:
        return 0.0

    dt = next_goal.elapsed_seconds - action.elapsed_seconds

    if dt < 0 or dt > epsilon:
        return 0.0

    if next_goal.team_id == action.team_id:
        return 1.0

    return -1.0
```

For neural-network training, map the target to `[0, 1]`:

```python
y_norm = (G + 1) / 2
```

Then at inference:

```python
shot_epv = 2 * y_hat_norm - 1
```

This matches the paper’s evaluation setup, where EPV targets in `[-1, 1]` are normalized to `[0, 1]` before MSE computation.

---

# 2. Baseline event-data xG feature

Before training the final shot EPV model, implement the **baseline xG model** used as an input feature.

This baseline xG is trained on a much larger event-data-only shot dataset. The paper uses **117,948 shots** and **12,266 goals** from OPTA event data. It uses only event-level information such as shot location, distance/angle to goal, attacking type, and whether the shot was a header. The model is an XGBoost classifier with grid search over number of trees, learning rate, and max depth; the best model used 100 trees, max depth 3, and learning rate (1e^{-1}).

## 2.1 Baseline xG features

Use:

```text
shot x-location
shot y-location
distance from shot to goal
angle from shot to goal
attacking type one-hot:
    open-play
    set-piece
    free-kick
    corner
    penalty
header flag:
    1 if shot taken with head
    0 otherwise
```

Target:

```text
1 if shot resulted in goal
0 otherwise
```

Model:

```text
XGBoost binary classifier
objective: binary logistic
metric: log loss
```

Grid search:

```python
n_estimators = [50, 100, 250]
learning_rate = [1e-3, 1e-2, 1e-1]
max_depth = [3, 5, 10]
```

Use 10-fold CV on the training set, standardize features, then save the calibrated prediction:

```python
baseline_xg = xgb_model.predict_proba(features)[:, 1]
```

You will feed `baseline_xg` into both:

```text
shot EPV model
action selection model
```

---

# 3. Shot EPV features

To match the paper, the final shot module should use scalar features, not a SoccerMap surface. The paper uses shallow neural networks for shot and ball-drive components, while SoccerMap is reserved for pass-related surfaces.

The shot EPV feature set should contain the features marked as **SE** in the appendix. The important ones are: ball location, distance/angle to goal, goalkeeper-related features, shot-blocking/interceptability features, baseline xG, and header flag.

Use this feature vector:

```text
1. ball x-location
2. angle between ball and opponent goal
3. distance between ball and opponent goal
4. boolean: ball is closer to the goal than the opponent goalkeeper
5. distance between ball and goalkeeper
6. y-axis distance between ball and goalkeeper
7. number of defending players inside the triangle formed by:
       ball location
       left goalpost
       right goalpost
8. number of defending players less than 3 meters from the ball
9. baseline event-data xG
10. header flag
```

The paper specifically highlights immediate pressure, shot interceptability/blockage count, and goalkeeper location as key shot features. It defines immediate pressure as opponents within 3 meters and interceptability as opponents inside the triangle between the shooter and the goalposts.

## 3.1 Goalkeeper features

You need reliable goalkeeper identification. Use, in order of preference:

```text
1. explicit position_name == "Goalkeeper" from player metadata
2. event/tracking provider role ID, if available
3. heuristic: defending player closest to own goal center over a stable window
```

For each shot:

```python
gk = defending_goalkeeper_at_frame

dist_ball_gk = euclidean_distance(ball_xy, gk_xy)
dist_ball_gk_y = abs(ball_y - gk_y)

ball_closer_than_gk = int(
    distance(ball_xy, opponent_goal_center)
    < distance(gk_xy, opponent_goal_center)
)
```

## 3.2 Interceptability / blockage count

Construct the triangle:

```text
P1 = ball location
P2 = left post of opponent goal
P3 = right post of opponent goal
```

Then count defenders inside it:

```python
block_count = sum(
    point_in_triangle(defender_xy, ball_xy, left_post, right_post)
    for defender in defending_players
)
```

## 3.3 Immediate pressure count

```python
pressure_count = sum(
    euclidean_distance(defender_xy, ball_xy) < 3.0
    for defender in defending_players
)
```

## 3.4 Shot model architecture

To match the paper’s reported parameter count, use a shallow MLP with:

```text
input dimension: 10
hidden layer 1: 10 units + ReLU
hidden layer 2: 10 units + ReLU
output layer: 1 unit + sigmoid
```

Parameter count:

```text
10*10 + 10 = 110
10*10 + 10 = 110
10*1 + 1   = 11
total      = 231
```

This matches the paper’s reported **231 parameters** for the Shot EPV model.

Architecture:

```python
class ShotEPVNet(nn.Module):
    def __init__(self, n_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

Training target:

```python
y_norm = (G + 1) / 2
```

Loss:

```python
loss = mse_loss(y_hat_norm, y_norm)
```

Inference:

```python
shot_epv = 2 * y_hat_norm - 1
```

## 3.5 Shot training setup

Use the same training protocol as the paper:

```text
optimizer: Adam
beta1: 0.9
beta2: 0.999
learning-rate grid: {1e-3, 1e-4, 1e-5, 1e-6}
batch-size grid: {16, 32}
early stopping delta for EPV models: 1e-5
split by match:
    train: 379 matches
    validation: 127 matches
    test: 127 matches
```

The paper reports the selected hyperparameters for Shot EPV as:

```text
batch size: 16
learning rate: 1e-3
loss: 0.2421
ECE: 0.0095
# parameters: 231
examples/s: 72,455
```



---

# 4. Action selection probability model

The action selection model estimates:

[
P(A=a \mid T_t)
]

for:

```text
A = pass
A = ball drive
A = shot
```

This is the model that tells the EPV framework how much weight to give to each action value in the current state. The paper says the final EPV is obtained by weighting each possible action’s expected value by the probability of taking that action in the given state.

## 4.1 Build the action selection dataset

Use all observed **open-play** examples from the three action classes:

```text
pass events
ball-drive events
shot events
```

Since you already implemented ball-drive modules, reuse the same processed ball-drive events. In the paper, ball drives longer than 1 second are split into 1-second ball drives, so the action selection model should be trained on that same processed action universe.

Each sample:

```text
input: tracking snapshot at action start
target: one-hot action type
```

Target encoding:

```python
pass       -> [1, 0, 0]
ball drive -> [0, 1, 0]
shot       -> [0, 0, 1]
```

The paper defines the action selection estimand as a **multinomially distributed random variable** with three possible values: pass, ball drive, or shot.

## 4.2 Action selection features

Use scalar state features, not surfaces. The appendix marks these as **AS** features. The model should include:

```text
1. ball x-location
2. angle between ball and opponent goal
3. distance between ball and opponent goal
4. attacking-team pitch control at the ball location
5. defending-team pitch influence at the ball location
6. index of closest attacking-team dynamic pressure line to ball
7. index of closest defending-team dynamic pressure line to ball
8. baseline event-data xG at the ball/action location
```

The paper says the action selection model uses ball location, distance/angle to goal, possession spatial information, team pitch control, opponent spatial influence near the ball, dynamic pressure lines, and baseline xG, especially to help model shot selection.

## 4.3 Action selection architecture

To match the paper’s reported **171 parameters**, use:

```text
input dimension: 8
hidden layer 1: 8 units + ReLU
hidden layer 2: 8 units + ReLU
output layer: 3 logits
softmax over logits
```

Parameter count:

```text
8*8 + 8 = 72
8*8 + 8 = 72
8*3 + 3 = 27
total    = 171
```

This matches the paper’s reported action selection model size.

Architecture:

```python
class ActionSelectionNet(nn.Module):
    def __init__(self, n_features: int = 8, n_actions: int = 3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(n_features, n_actions)

    def forward(self, x):
        logits = self.classifier(self.feature_extractor(x))
        return logits
```

Training loss:

```python
loss = cross_entropy(logits, class_index)
```

Do **not** apply softmax before `nn.CrossEntropyLoss`; PyTorch does it internally.

At inference:

```python
probs = torch.softmax(logits, dim=-1)

p_pass = probs[:, 0]
p_drive = probs[:, 1]
p_shot = probs[:, 2]
```

## 4.4 Action selection training setup

Use the same general training setup:

```text
optimizer: Adam
beta1: 0.9
beta2: 0.999
learning-rate grid: {1e-3, 1e-4, 1e-5, 1e-6}
batch-size grid: {16, 32}
early stopping delta: 1e-3
split by match, not by event
shuffle events in the training set
```

The paper reports:

```text
batch size: 32
learning rate: 1e-3
loss: 0.6454
# parameters: 171
examples/s: 23,709
```



---

# 5. Integration into the final EPV computation

Once the shot and action selection models are trained, your final inference pipeline for a frame/state (T_t) should be:

```python
# pass modules, already implemented
pass_value_surface = compute_pass_epv_surface(T_t)
pass_selection_surface = compute_pass_selection_surface(T_t)

pass_value = sum_over_pitch(
    pass_selection_surface[l] * pass_value_surface[l]
)

# ball-drive modules, already implemented
drive_value = compute_ball_drive_epv(T_t)

# new shot module
shot_value = shot_epv_model(shot_features(T_t))

# new action selection model
p_pass, p_drive, p_shot = action_selection_model(action_features(T_t))

# final EPV
epv = (
    p_pass * pass_value
    + p_drive * drive_value
    + p_shot * shot_value
)
```

Conceptually:

[
EPV(T_t)
========

P(A=\rho \mid T_t)V_{\text{pass}}(T_t)
+
P(A=\delta \mid T_t)V_{\text{drive}}(T_t)
+
P(A=\varsigma \mid T_t)V_{\text{shot}}(T_t)
]

where the pass value itself is an expectation over every possible destination location.

---

# 6. Implementation order I would follow

## Step 1 — Freeze your existing pass and ball-drive modules

Before adding shot/action selection, freeze and test:

```text
pass success probability surface
pass successful EPV surface
pass missed EPV surface
pass destination selection surface
ball-drive success probability
ball-drive successful EPV
ball-drive missed EPV
```

Make sure all outputs are in the same EPV scale:

```text
[-1, 1]
```

## Step 2 — Implement the baseline event xG model

Train XGBoost on event shots.

Save:

```text
baseline_xg_model.pkl
baseline_xg_feature_scaler.pkl
```

You will need this during both shot EPV and action selection feature generation.

## Step 3 — Generate shot EPV dataset

For every open-play shot:

```text
extract tracking snapshot
normalize left-to-right
compute 10 shot features
compute long-term reward G
store y_norm = (G + 1) / 2
```

Suggested artifact:

```text
shot_epv_dataset.parquet
```

Columns:

```text
match_id
event_id
frame_id
period
elapsed_seconds
team_id
player_id
ball_x_norm
angle_ball_goal
distance_ball_goal
ball_closer_than_gk
distance_ball_gk
distance_ball_gk_y
block_count
pressure_count_3m
baseline_xg
is_header
reward_G
reward_norm
split
```

## Step 4 — Train ShotEPVNet

Use the exact shallow MLP:

```text
10 -> 10 -> 10 -> 1
```

Loss:

```text
MSE on normalized reward
```

Validation:

```text
MSE
ECE with 10 quantile bins
calibration plot
```

## Step 5 — Generate action selection dataset

For every open-play pass, ball-drive, and shot:

```text
extract tracking snapshot
normalize left-to-right
compute 8 action-selection features
assign class label
```

Suggested artifact:

```text
action_selection_dataset.parquet
```

Columns:

```text
match_id
event_id
frame_id
period
elapsed_seconds
team_id
player_id
action_type
action_class_index
ball_x_norm
angle_ball_goal
distance_ball_goal
pitch_control_attacking_at_ball
pitch_influence_defending_at_ball
closest_attacking_line_to_ball
closest_defending_line_to_ball
baseline_xg
split
```

## Step 6 — Train ActionSelectionNet

Architecture:

```text
8 -> 8 -> 8 -> 3
```

Loss:

```text
categorical cross-entropy
```

Validation:

```text
cross-entropy
top-1 accuracy, optional
per-class reliability plots, optional
mean predicted action probabilities by field zone
```

The paper uses SHAP to inspect which features drive action selection and reports that pressure/pitch-control features are important for pass vs ball-drive choice, while distance and angle to goal dominate shot selection.

## Step 7 — Compose the final EPV

For each frame where the ball is controlled by a player:

```text
compute pass value
compute ball-drive value
compute shot value
compute action probabilities
combine
```

For visualization, expose:

```text
P(pass | T)
P(ball drive | T)
P(shot | T)
V_pass(T)
V_drive(T)
V_shot(T)
EPV(T)
```

This mirrors the control-room style interpretation in the paper, where the final display shows the instantaneous action selection probabilities, each action’s expected value, and the final EPV.

---

# 7. Important pitfalls

The biggest mistake would be to train the shot module as only:

```text
shot -> goal / no goal
```

That is only the **baseline xG** part. The final shot EPV module should learn the long-term possession reward for shot actions.

Second, do not train the action selection model on only event rows if your ball-drive module created 1-second ball-drive chunks. The action selection model must see the same action universe used by the EPV decomposition:

```text
passes
1-second ball drives
shots
```

Third, split by **match**, not by event. Otherwise, adjacent events from the same possession may leak across train/validation/test.

Fourth, keep all models on a consistent orientation and scale. The paper’s features assume the attacking team always attacks left to right.

My practical recommendation: implement the baseline xG first, then the shot module, then the action selection model, and only after that integrate final EPV. The action selection model depends on the baseline xG feature, but not on the final shot EPV model itself.

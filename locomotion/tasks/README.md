# Task Hyperparameters

This folder contains the task definitions and the task-side JSON sweep files:

- `bittle_walk_env.py`
- `bittle_walking_hparams.json`
- `bittle_dance_env.py`
- `bittle_dance_hparams.json`

These JSON files are not the PPO trainer settings. They are the environment-side settings: how strongly the robot is rewarded for certain behavior, how aggressive its actions can be, how noisy the world is, and so on.

The trainer-side sweep file lives elsewhere:

- `locomotion/sweeps/training_budget_and_batching_sweep.json`

That trainer-side file controls things like:

- how many total steps to train
- how many environments run in parallel
- batch size
- minibatching

This README only covers the task-side hyperparameters in `locomotion/tasks/`.

## How To Read The Tables

- `Default` is the value used when the JSON entry does not override it.
- `Higher` means numerically larger.
- `Lower` means numerically smaller.

For penalty scales, that matters:

- `-0.5` is higher than `-1.0`
- so a higher negative penalty usually means a weaker punishment
- and a lower negative penalty usually means a stronger punishment

An empty JSON object like `{}` means "use the task defaults exactly as written in the environment code."

## Walking

The walking task means:

- walk forward or backward at the requested speed
- drift left or right if asked
- turn left or right at the requested rate
- stay upright and avoid sloppy motion while doing it

The walking task JSON file is:

- `locomotion/tasks/bittle_walking_hparams.json`

### Walking Hyperparameters

| Hyperparameter | Default | What it controls | Higher value | Lower value |
| --- | ---: | --- | --- | --- |
| `obs_noise` | `0.05` | How much random fuzz is added to what the policy "senses." | Harder but often more robust to imperfect observations. | Easier and cleaner training, but can be more brittle. |
| `action_scale` | `0.5` | How far each action can push joints away from the default pose. | Bigger, more aggressive leg motions; easier to overreact or tip. | Gentler, safer motions; can become too weak or stiff. |
| `kick_vel` | `0.05` | How strong the random body shoves are when kicks are enabled. | Stronger disturbance training; tougher but more robust. | Gentler disturbances; easier but less recovery practice. |
| `enable_kicks` | `true` | Whether random shoves happen at all. | `true`: teaches recovery from bumps. | `false`: no random bump recovery training. |
| `n_frames` | `5` | How many tiny MuJoCo physics steps happen per one policy action. | More simulator work per action; smoother/more resolved physics, slower runs. | Fewer physics substeps; faster runs, rougher action timing. |
| `tracking_lin_vel_scale` | `2.5` | How strongly the robot is rewarded for matching the requested forward/sideways speed. | Makes speed-following more important than comfort or smoothness. | Makes exact pace-following less important. |
| `tracking_ang_vel_scale` | `1.5` | How strongly the robot is rewarded for matching the requested turning rate. | Stronger incentive to turn exactly as asked. | Weaker incentive to turn accurately. |
| `lin_vel_z_scale` | `-2.0` | Penalty for bouncing upward and downward too fast. | Less punishment for vertical bobbing. | More punishment for vertical bobbing. |
| `ang_vel_xy_scale` | `-0.05` | Penalty for unwanted roll/pitch wobble speed. | More wobble is tolerated. | Wobble is punished more strongly. |
| `orientation_scale` | `-5.0` | Penalty for leaning away from upright. | More tilt is tolerated. | Staying upright matters more. |
| `torques_scale` | `-0.0002` | Penalty for pushing the motors hard. | Stronger pushes are allowed more freely. | Encourages gentler, lower-effort motion. |
| `action_rate_scale` | `-0.001` | Penalty for changing commands too abruptly from one step to the next. | Jerkier command changes are tolerated more. | Motion becomes smoother but less snappy. |
| `joint_acc_scale` | `-0.0025` | Penalty for very abrupt changes in joint speed. | Fast snaps are tolerated more. | Stronger pressure for smooth leg motion. |
| `stand_still_scale` | `-0.5` | Penalty for moving joints when the command says "basically stand still." | The robot is allowed to fidget more when it should stay put. | Standing quietly matters more. |
| `termination_scale` | `-1.0` | Penalty for ending the episode early by failing. | Falling early hurts less. | Falling early hurts more. |
| `feet_air_time_scale` | `1.0` | Reward for reasonable swing timing when the robot is commanded to move. | More encouragement for stepping rhythm. | Less encouragement for that stepping rhythm. |
| `foot_slip_scale` | `-0.04` | Penalty for feet sliding sideways while touching the ground. | Sliding is punished less. | Sliding is punished more. |
| `energy_scale` | `-0.002` | Penalty for wasting energy. | Wasteful motion is tolerated more. | Efficiency matters more. |

## Dancing

The dance task means:

- stay mostly in place
- follow a repeating rhythmic pose pattern
- keep a small torso bounce with the beat
- stay upright and avoid collapsing early

The dance task JSON file is:

- `locomotion/tasks/bittle_dance_hparams.json`

### Dance Hyperparameters

| Hyperparameter | Default | What it controls | Higher value | Lower value |
| --- | ---: | --- | --- | --- |
| `obs_noise` | `0.05` | How much random fuzz is added to the observations. | Harder but usually more robust. | Easier and cleaner, but less robust. |
| `action_scale` | `0.5` | How far one action can push joints away from the default pose. | Bigger, punchier dance moves; easier to destabilize. | Smaller, safer motions; can become too timid. |
| `kick_vel` | `0.03` | How strong the random shoves are when kicks are enabled. | Harder recovery training. | Easier, gentler disturbance training. |
| `enable_kicks` | `true` | Whether those random shoves happen. | `true`: trains recovery from bumps. | `false`: no bump recovery pressure. |
| `cycle_steps` | `100` | How many control steps make up one full dance phrase. | Slower, more drawn-out dance cycle. | Faster, more rapid dance cycle. |
| `dance_amplitude` | `1.0` | How large the scripted target dance motions are. | Bigger pose swings and larger bounce. | Smaller, tighter dance motions. |
| `n_frames` | `5` | How many tiny physics steps happen for each policy action. | More resolved physics, slower runs. | Less resolved physics, faster runs. |
| `pose_tracking_scale` | `4.0` | How strongly the robot is rewarded for matching the target joint pose for the current beat. | Matching the scripted pose matters more. | The pose target matters less. |
| `bounce_tracking_scale` | `1.5` | How strongly the torso is rewarded for matching the target up-and-down bounce. | The torso bob matters more. | The bounce matters less. |
| `upright_scale` | `1.25` | How strongly the robot is rewarded for staying upright. | Staying upright matters more. | The policy can sacrifice uprightness more easily. |
| `stay_centered_scale` | `1.0` | How strongly the robot is rewarded for staying near its starting spot. | Drifting away is discouraged more. | The robot is allowed to wander more. |
| `foot_stability_scale` | `0.6` | How much the robot is rewarded for keeping a stable number of feet in contact. | More pressure for stable support under the dance. | Less pressure for foot-contact stability. |
| `action_rate_scale` | `-0.002` | Penalty for sudden command jumps from one step to the next. | Jerky command changes are tolerated more. | Smoother command changes are encouraged more. |
| `joint_acc_scale` | `-0.003` | Penalty for abrupt changes in joint speed. | Snappier motion is tolerated more. | Smoothness matters more. |
| `torques_scale` | `-0.0002` | Penalty for pushing the motors too hard. | More forceful motion is tolerated more. | Lower-effort motion is preferred more strongly. |
| `energy_scale` | `-0.002` | Penalty for wasting energy. | Wasteful motion is tolerated more. | Efficient motion matters more. |
| `termination_scale` | `-20.0` | Penalty for failing and ending the dance early. | Early failure hurts less. | Early failure hurts more. |

## Practical Advice

- If the robot looks too violent or twitchy, try lowering `action_scale` or making `action_rate_scale` and `joint_acc_scale` more negative.
- If the robot ignores the task goal, increase the main positive reward for that goal:
  - `tracking_lin_vel_scale` and `tracking_ang_vel_scale` for walking
  - `pose_tracking_scale`, `bounce_tracking_scale`, `upright_scale`, or `stay_centered_scale` for dancing
- If the robot keeps falling quickly, make `termination_scale` lower so failure hurts more.
- If training is good in simulation but fragile, try enabling kicks and/or raising `kick_vel` a little.

## Relationship To The Sweep Runner

The sweep runner loops over:

1. trainer-side entries from `locomotion/sweeps/training_budget_and_batching_sweep.json`
2. task-side entries from the matching JSON in this folder

So one final trial is:

- one trainer override bundle
- plus one task override bundle

That is why each sweep trial folder contains:

- `training_overrides.json`
- `task_overrides.json`
- `combined_overrides.json`

# Trainer-Side Sweep Parameters

This folder contains the trainer-side sweep configuration:

- `training_budget_and_batching_sweep.json`
- `hparam_sweep.py`

These settings control how PPO training is run. They are different from the task-side settings under `locomotion/tasks/`.

In plain language:

- `locomotion/sweeps/*.json` controls the training budget and how PPO slices that training data up.
- `locomotion/tasks/*.json` controls the task world itself and what behavior is rewarded.

## What This JSON Represents

`training_budget_and_batching_sweep.json` is a list of trainer override bundles.

Each object is one trainer-side configuration, for example:

```json
{
  "num_timesteps": 200000,
  "num_evals": 6,
  "episode_length": 200,
  "num_envs": 32,
  "batch_size": 32,
  "unroll_length": 8,
  "num_minibatches": 4
}
```

That trainer bundle is then paired with one task-side bundle from `locomotion/tasks/`, so one final trial is:

- one trainer override bundle
- plus one task override bundle

## How The Loop Actually Works

The sweep is a nested loop.

In plain English, it does this:

1. Read the list of training-budget entries from `training_budget_and_batching_sweep.json`.
2. Pick the first training-budget entry.
3. While holding that one fixed, loop through every task-hyperparameter entry from the selected task JSON.
4. After it finishes all task entries for that one training-budget entry, move to the next training-budget entry.
5. Repeat until every outer entry has been paired with every inner entry.

So the real loop order is:

1. outer loop = trainer-side JSON
2. inner loop = task-side JSON

That means each row in `training_budget_and_batching_sweep.json` acts like a "main bucket" of training settings.
Inside that bucket, the sweep then tries every matching task-world variation.

## Where The Values Come From

The sweep values are mainly coming from the JSON files themselves.

That means:

- the training-budget JSON tells the sweep which trainer-side cases to try
- the task JSON tells the sweep which task-side cases to try

If a JSON entry leaves a field out, the code falls back to the normal built-in default for that missing field.
But the important practical point is: the sweep is driven by the list of cases written in the JSON files.

An empty object like `{}` means:

- "run one trial using the normal built-in defaults for this side of the sweep"

So:

- `{}` in the trainer-side JSON means "use the normal trainer defaults"
- `{}` in the task-side JSON means "use the normal task defaults"

## Mental Model

The easiest way to think about these knobs is:

- `num_timesteps` = total amount of practice
- `episode_length` = how long one attempt can last
- `num_envs` = how many robot copies practice at the same time
- `unroll_length` = how long each rollout snippet is
- `batch_size` = how many snippets PPO averages in one optimizer chunk
- `num_minibatches` = how many chunks PPO splits the collected data into
- `num_updates_per_batch` = how many times PPO reuses the same collected data before gathering more

## Parameter Reference

| Hyperparameter | Default in current dance sweep | What it controls | Higher value | Lower value |
| --- | ---: | --- | --- | --- |
| `num_timesteps` | varies: `150000`, `200000`, `250000` | Total amount of practice before training stops. | Longer training, more chances to improve, more time spent. | Faster runs, less total learning opportunity. |
| `num_evals` | `6` | How many progress check-ins happen during training. | More report-card checkpoints and a more detailed final curve, but more overhead. | Fewer checkpoints and less reporting overhead. |
| `episode_length` | `200` | Maximum length of one attempt before it resets. | Longer attempts and more time to recover inside one episode. | Quicker resets and shorter attempts. |
| `num_envs` | varies: `24`, `32`, `48` | How many simulated robots practice in parallel. | More data gathered at once; more memory use; different optimization behavior. | Less parallel data; lighter runs; slower data collection. |
| `batch_size` | varies: `24`, `32`, `48`, `64` | How many rollout snippets PPO averages in one optimizer chunk. | Smoother, less noisy updates; heavier optimizer work; can become less responsive. | Noisier, lighter, more reactive updates; can become unstable. |
| `unroll_length` | `8` | How many consecutive steps go into one rollout snippet. | Longer cause-and-effect window; more memory use; slower update cadence. | Shorter snippets; lighter/faster; weaker delayed-effect signal. |
| `num_minibatches` | `4` | How many smaller optimizer slices the collected data is split into. | More slicing of the collected rollout data. | Fewer, larger slices. |
| `num_updates_per_batch` | `1` | How many optimization passes PPO makes before collecting fresh data again. | Reuses the same batch more; more compute per batch; can overfit stale rollouts. | Uses fresher data sooner; less reuse. |

## What The Current JSON Is Actually Sweeping

The current `training_budget_and_batching_sweep.json` is mainly exploring:

- total training budget via `num_timesteps`
- parallel data collection scale via `num_envs`
- optimizer chunk size via `batch_size`

It is currently holding these steady:

- `num_evals = 6`
- `episode_length = 200`
- `unroll_length = 8`
- `num_minibatches = 4`

So the current trainer-side sweep is mostly asking:

- how long should the task train?
- how many robot copies should practice in parallel?
- how big should each PPO learning chunk be?

Then, for each one of those trainer-side cases, the inner loop tries every task-side case from the selected task JSON.

## Practical Advice

- If runs are too slow or too expensive, lower `num_timesteps` or `num_envs`.
- If PPO looks noisy and unstable, try increasing `batch_size`.
- If PPO looks too sluggish or over-averaged, try decreasing `batch_size`.
- If the task needs a longer cause-and-effect window inside each training example, increase `unroll_length`.
- If you change `num_envs`, keep an eye on GPU memory and on whether learning behavior changes sharply.

## Relationship To `Scripts/sweep.sh`

`Scripts/sweep.sh` now has one obvious task switch at the very top:

- `CURRENT_TASK="Dance"` or `CURRENT_TASK="Walk"`

That selected task then decides:

- which task environment file is being targeted
- which task-side JSON is used for the inner loop
- which `--task` value is passed to `hparam_sweep.py`

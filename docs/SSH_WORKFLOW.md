# Pupper Simulations SSH Workflow

Document detailing the iterative process on simulating reinforcement learning policies on a remote SSH server.

## Setup

Talk to a Triton Droids lead for instructions on how to obtain a SSH key for the Triton Droids server

1. Connect to the remote server

```bash
ssh -i ~/.ssh/{your_ssh_key_here} tritondroids@132.249.64.152
```

2. Set up [PORT FORWARDING](PORT_FORWARDING.md)

## Workflow

1. Pull remote changes from github (switch to your branch if needed)

```bash
cd orengershony/pupper-simulations/locomotion
git pull
```

2. Use the UV virtual environment to run the training script (see [train.py](../locomotion/train.py), [training_config.py](../locomotion/training_config.py), and [training_helpers.py](../locomotion/training_helpers.py) for how to set up system arguments and custom configs)

```bash
uv run train.py
```

3. Refresh your localhost (http://localhost:8000/) to see updated output files. To edit the training script, make changes on your local machine (ON YOUR BRANCH ONLY) and push the changes to the github repository.

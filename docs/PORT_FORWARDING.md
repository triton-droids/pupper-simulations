# Port Forwarding Setup

Instructions on how to set up and use port forwarding in the ssh workflow

## Setup

1. In a separate terminal, SSH into the remote server

```bash
ssh -i ~/.ssh/{your_ssh_key_here} tritondroids@132.249.64.152
```

2. Navigate to the pupper simulations directory

```bash
cd orengershony/pupper-simulations
```

3. Start a simple python HTTP server

```bash
uv run python -m http.server 8000
```

4. On a separate terminal on your LOCAL MACHINE, set up SSH port forwarding

```bash
ssh -L 8000:localhost:8000 -i ~/.ssh/{your_ssh_key_here} tritondroids@132.249.64.152
```

5. Now at http://localhost:8000/ on your local machine you should be able to see the remote file system

# Development

First of all, thank you for your interest in using and contributing to **daspal**.
Whether it's fixing a bug, improving the documentation, or proposing new features, your help is very welcome.

This guide will help you get started with setting up a development environment and contributing effectively.

### Clone the repository

Clone the package in a specific directory (ex: /home/username/py_code/daspal) which defines the **path** to the package

```console
git clone -b devel https://github.com/daspal-project/daspal.git
cd daspal
```

### Create a Python environment

#### Recommended (modern setup)

All dependencies then handled with pip

```bash
conda create -n daspal python=3.10
conda activate daspal
```

#### Alternative (older systems)

For compatibility with glibc, conda YAML ensures C dependencies are precompiled.

```bash
conda env create -f daspal.yml
conda activate daspal
```

### Install Daspal

#### Editable install (recommended)

```bash
pip install -e .
```

This mode allows you to modify the code and see changes immediately without reinstalling.

#### Optional dependencies

* Local notebooks (with jupyterlab):

Installs dependencies needed to run notebooks locally

```bash
pip install .[notebook]
```

* Server / JupyterHub (Dask, remote notebooks):

Installs dependencies needed for running notebooks on a JupyterHub server

```bash
pip install .[notebook_server]
```

### Dask dashboard (server setup)

#### Jupyterhub dashboard link

Add to `~/.config/dask/distributed.yaml`:

```yaml
distributed:
  dashboard:
    link: "{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status"
```

#### Running a python script on a server (e.g. via a SLURM job)

Add the following in the script

```python
from daspal.dask_utils.dashboard import log_dask_dashboard_info

log_dask_dashboard_info(
    client,
    cluster="server_address",
    logfile=f"scheduler_{os.environ.get('SLURM_JOB_ID', 1)}.log"
)
```

Then in the file `scheduler_jobID.log` find the following command for port forwarding (e.g. port 8787) and run it on your local machine

```console
ssh -N -L 8787:nodeID:8787 username@server
```

The dask dashboard can then be accessed at http://localhost:8787 using a browser

### Contributing and building the documentation

The documentation consists of two parts: the docstrings in the code itself and the docs in this folder `daspal/docs/`.

Install the package with the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML pages:

```bash
cd docs/
make html
```

To view them at http://localhost:8000:

```bash
python -m http.server 8000 -d _build/html
```

Stop the server with `Ctrl+C`.


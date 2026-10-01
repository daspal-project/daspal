import os
from dask.distributed import Client

def log_dask_dashboard_info(client: Client, cluster: str = "server", logfile: str = "scheduler.log"):
    '''
    When running dask on HPC the function logs the Dask dashboard port
    and a ready-to-use SSH tunneling command to access the dashboard from the local machine.
    
    Args:
        client: Dask Client object
        cluster: name of HPC cluster
        logfile: Path to the log file to write the information
    '''
    port = client.scheduler_info()["services"]["dashboard"]
    ssh_command = f'ssh -N -L {port}:{os.environ["HOSTNAME"]}:{port} {os.environ["USER"]}@{cluster}'

    # Write to logfile
    with open(logfile, "w") as f:
        print("Dask scheduler node:", os.environ["HOSTNAME"], file=f, flush=True) # flush ensures that each print is immediately written, no buffer
        print(f"Command for dashboard port forwarding:", file=f, flush=True)
        print(ssh_command, file=f, flush=True)
        print(f"On local machine open dashboard at: http://localhost:{port}", file=f, flush=True)
        print("-" * 60, file=f, flush=True)

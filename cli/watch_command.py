import click

from core.job_watcher import JobWatcher


@click.command()
def watch():
    input_path = "data/input"
    output_path = "data/output"

    watcher = JobWatcher(output_path)

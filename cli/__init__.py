import logging


import click

from cli.watch_command import watch
from cli.create_job import create_job

@click.group()
def cli():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

cli.add_command(watch)
cli.add_command(create_job)
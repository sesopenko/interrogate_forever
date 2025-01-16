import logging


import click

from watch_command import watch

@click.group()
def cli():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

cli.add_command(watch)
import logging
import os

import click
from watchdog.observers import Observer

from core.job_watcher import InputObserver
from core.input_watcher import InputWatcher
from core.interrogator import Interrogator

logger = logging.getLogger(__name__)

@click.command()
def watch():
    input_path = os.path.join(os.getcwd(), 'data', 'input')
    output_path = os.path.join(os.getcwd(), 'data', 'output')
    working_path = os.path.join(os.getcwd(), 'data', 'working')

    interrogator = Interrogator()
    watcher = InputWatcher(output_path, working_path, interrogator)
    watcher.clean_start()
    observer = Observer()
    input_observer = InputObserver(input_path, observer, watcher)
    logging.info("Starting input observer")
    input_observer.start()
import logging
import os
import time

from watchdog.observers import Observer

from core.input_watcher import InputWatcher

logger = logging.getLogger(__name__)


class InputObserver():
    def __init__(self, input_path: str, observer: Observer, output_watcher: InputWatcher):
        self._observer = observer
        self._input_path: str = input_path
        self._output_watcher: InputWatcher = output_watcher
    def start(self):
        os.makedirs(self._input_path, exist_ok=True)
        self._observer.schedule(self._output_watcher, self._input_path, recursive=False)
        self._observer.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._observer.stop()
        # wait until it terminates:
        self._observer.join()

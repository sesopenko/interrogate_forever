import logging
from datetime import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class JobWatcher(FileSystemEventHandler):
    def __init__(self, output_path: str):
        self._output_path = output_path
        pass
    def on_created(self, event):
        print(f"File created: {event.src_path}")

class InputObserver():
    def __init__(self, input_path: str, observer: Observer, watcher: JobWatcher):
        self._observer = observer
        self._input_path: str = input_path
        self._watcher: JobWatcher = watcher
    def start(self):
        self._observer.schedule(self._watcher, self._input_path, recursive=False)
        self._observer.start()

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._observer.stop()
        # wait until it terminates:
        self._observer.join()
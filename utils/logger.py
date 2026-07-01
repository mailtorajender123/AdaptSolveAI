import json
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.records = []

    def log(self, record: dict) -> None:
        record = dict(record)
        record['time'] = datetime.now().isoformat(timespec='seconds')
        self.records.append(record)
        print(json.dumps(record, indent=None, default=str))

    def save(self, filename: str = 'log.json') -> str:
        path = os.path.join(self.out_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)
        return path

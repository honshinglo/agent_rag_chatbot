import random
import time

def get_current_wait_time(worker: str) -> int | str:

    worker = worker.lower()

    if worker not in ["marco", "louis", "mary"]:
        return f"Worker {worker} does not exist"

    time.sleep(1)

    return random.randint(0, 100)
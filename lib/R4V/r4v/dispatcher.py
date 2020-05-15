import multiprocessing as mp
import os
import psutil
import time

from . import logging


def get_memory_usage():
    try:
        p = psutil.Process(os.getpid())
        children = p.children(recursive=True)
        memory = 0
        for child in children:
            memory += child.memory_info().rss
    except psutil.NoSuchProcess:
        memory = 0
    return memory


def dispatch(target, args, max_memory=-1, timeout=-1):
    logger = logging.getLogger(__name__)
    proc = mp.Process(target=target, args=args)

    start_t = time.time()
    proc.start()

    try:
        while proc.is_alive():
            time.sleep(0.01)
            now_t = time.time()
            duration_t = now_t - start_t
            mem_usage = get_memory_usage()
            if max_memory >= 0 and mem_usage >= max_memory:
                logger.error(
                    "Out of Memory (killing process): %d > %d", mem_usage, max_memory
                )
                proc.terminate()
                break
            if timeout >= 0 and duration_t >= timeout:
                logger.error("Timeout (killing process): %d > %d", duration_t, timeout)
                proc.terminate()
                break
        else:
            logger.info("Process finished successfully.")
    except KeyboardInterrupt:
        logger.error("Received keyboard interupt (killing process)")
        proc.terminate()


#!/usr/bin/env python

# TRIFLE - UTILITY FUNCTIONS
# Version: 17-06-2025 by TJ de Kloe

import psutil
import os
from datetime import datetime

def log_memory_usage(stage, memlog_path=None):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {stage} – Memory usage: {mem_mb:.2f} MB"

    # Print and log if logging is set up
    import logging
    logging.info(message)

    # Also write to separate memory log if path provided
    if memlog_path:
        with open(memlog_path, "a") as f:
            f.write(message + "\n")
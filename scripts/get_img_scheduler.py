import subprocess
import time
from datetime import datetime, timedelta

interval = 60
run_duration = timedelta(minutes=3)

start_time = datetime.now()
while datetime.now() - start_time < run_duration:
    cycle_start = datetime.now()
    subprocess.run(['python', 'scripts/get_imgs.py'])
    cycle_end = datetime.now()

    if (cycle_end - cycle_start).seconds < interval:
        time.sleep(interval - (cycle_end - cycle_start).seconds)
from collections import deque
import time
from typing import Any, Callable


class Throttler:
    def __init__(self, max_req: int, period: float, time_func: Callable[[], float] = time.time) -> None:
        self._max_req = max_req
        self._period = period
        self._request_timestamps = deque(maxlen=max_req)
        self._req_cnt = 0
        self._time_func = time_func

    def throttle(self, val: Any) -> Any:
        t1 = self._time_func()
        self._req_cnt += 1
        self._request_timestamps.append(t1)
        if len(self._request_timestamps) >= self._max_req:
            t0 = self._request_timestamps[0]
            dt = t1 - t0
            if dt < self._period:
                time.sleep(self._period - dt)
        return val

from unittest import TestCase
from unittest.mock import patch, MagicMock, call

from pyxtension.streams import stream
from pyxtension.throttler import Throttler


class TestThrottler(TestCase):
    @patch('time.sleep', return_value=None)
    def test_throttle(self, sleep_mock):
        time_func = MagicMock()
        time_func.side_effect = [1, 1, 1, 6, 6, 8, 9, 11, 12, 13]
        throttler = Throttler(2, 3, time_func)  # max 2 requests per 3 seconds
        lst = stream(range(10)).map(throttler.throttle).to_list()
        self.assertListEqual(sleep_mock.call_args_list,
                             [call(3), call(3), call(3), call(1), call(2), call(1), call(2), call(2)])
        self.assertEqual(lst, list(range(10)))

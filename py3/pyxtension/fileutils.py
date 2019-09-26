#!/usr/bin/python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: utility library
import bz2
import gzip
import lzma
import sys
import time

from bz2 import BZ2File
from collections import OrderedDict
from gzip import GzipFile
from math import floor, log10
from typing import Optional, Callable, TextIO, Union, BinaryIO, TypeVar

K = TypeVar('K')

__author__ = 'ASU'


def openByExtension(filename: str, mode: str = 'r', buffering: int = -1,
                    compresslevel: int = 9, **kwargs) -> Union[TextIO, BinaryIO, GzipFile, BZ2File]:
    """
    :return: Returns an opened file-like object, decompressing/compressing data depending on file extension
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, compresslevel=compresslevel, **kwargs)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode, compresslevel=compresslevel, **kwargs)
    elif filename.endswith('.xz'):
        my_filters = [
            {"id": lzma.FILTER_LZMA2, "preset": compresslevel | lzma.PRESET_EXTREME}
        ]
        return lzma.open(filename, mode=mode, filters=my_filters, **kwargs)
    else:
        return open(filename, mode, buffering=buffering, **kwargs)


open_by_ext = openByExtension
smart_open = openByExtension


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        interval: Minimum visual progress update interval (in seconds).
    This class was inspired from keras.utils.Progbar
    """

    def __init__(self, target: Optional[int],
                 width: int = 30,
                 verbose: bool = False,
                 interval: float = 0.5,
                 stdout: TextIO = sys.stdout,
                 timer: Callable[[], float] = time.time,
                 dynamic_display: Optional[bool] = None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.stdout = stdout
        if dynamic_display is None:
            self._dynamic_display = ((hasattr(self.stdout, 'isatty') and self.stdout.isatty())
                                     or 'ipykernel' in sys.modules
                                     or (hasattr(self.stdout, 'name')
                                         and self.stdout.name in ('<stdout>', '<stderr>'))
                                     )
        else:
            self._dynamic_display = dynamic_display
        self._total_width = 0
        self._seen_so_far = 0
        self._values = OrderedDict()
        self._timer = timer
        self._start = self._timer()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
        """
        values = values or []
        for k, v in values:
            if k not in self._values:
                self._values[k] = [v * (current - self._seen_so_far),
                                   current - self._seen_so_far]
            else:
                self._values[k][0] += v * (current - self._seen_so_far)
                self._values[k][1] += (current - self._seen_so_far)
        self._seen_so_far = current

        now = self._timer()
        info = ' - %.0fs' % (now - self._start)
        if (now - self._last_update < self.interval and
                self.target is not None and current < self.target):
            return

        prev_total_width = self._total_width

        if self.target is not None:
            numdigits = int(floor(log10(self.target))) + 1
            barstr = '%%%dd/%d [' % (numdigits, self.target)
            bar = barstr % current
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
        else:
            bar = '%7d/Unknown' % current

        if current:
            time_per_unit = (now - self._start) / current
        else:
            time_per_unit = 0
        if self.target is not None and current < self.target:
            eta = time_per_unit * (self.target - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = ' - ETA: %s' % eta_format
        else:
            if time_per_unit >= 1:
                info += ' %.0fs/step' % time_per_unit
            elif time_per_unit >= 1e-3:
                info += ' %.0fms/step' % (time_per_unit * 1e3)
            else:
                info += ' %.0fus/step' % (time_per_unit * 1e6)

        for k in self._values:
            info += ' - %s:' % k
            if isinstance(self._values[k], list):
                avg = self._values[k][0] / max(1, self._values[k][1])
                # avg = mean(        )
                if abs(avg) > 1e-3:
                    info += ' %.3f' % avg
                else:
                    info += ' %.3e' % avg
            else:
                info += ' %s' % self._values[k]

        self._total_width += len(info)
        if prev_total_width > self._total_width:
            info += (' ' * (prev_total_width - self._total_width))

        display_str = bar + info

        if self._dynamic_display:
            prev_total_width = self._total_width
            self._total_width = len(display_str)
            # ASU: if \r doesn't work, use \b - to move cursor one char back
            display_str = '\r' + display_str + ' ' * max(0, prev_total_width - len(display_str))
        else:
            display_str = display_str + '\n'
        if self.target is not None and current >= self.target:
            display_str += '\n'
        self.stdout.write(display_str)
        self.stdout.flush()

        if self.verbose:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = self._values[k][0] / max(1, self._values[k][1])
                    # avg = mean()
                    if avg > 1e-3:
                        info += ' %.3f' % avg
                    else:
                        info += ' %.3e' % avg

                display_str = info
                if self._dynamic_display:
                    prev_total_width = self._total_width
                    self._total_width = len(display_str)
                    # ASU: if \r doesn't work, use \b - to move cursor one char back
                    display_str = '\r' + display_str + ' ' * max(0, prev_total_width - len(display_str))
                else:
                    display_str = display_str + '\n'
                self.stdout.write(display_str)
                self.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def __call__(self, el: K) -> K:
        """
        It's intended to be used from a mapper over a stream of values.
        It returns the same el
        # Example:
        >>> from pyxtension.fileutils import Progbar
        >>> stream(range(3)).map(Progbar(3)).size()
        1/3 [=========>....................] - ETA: 0s
        2/3 [===================>..........] - ETA: 0s
        3/3 [==============================] - 0s 100ms/step
        """
        self.add(1, None)
        return el

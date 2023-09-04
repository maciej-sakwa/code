"""
Own implementation of the IVI classes TraceY and TraceYT.
"""

import numpy as np
import copy
from typing import TypeVar
from warnings import warn

# noinspection PyTypeChecker
T = TypeVar('T', bound='TraceY')


class TraceY(object):
    """Y trace object"""

    def __init__(self):
        self.average_count = 1
        self.y_increment = 0
        self.y_origin = 0
        self.y_reference = 0
        self.y_raw = None
        self.y_hole = None

        # indicate a single selected sample
        self._single_sample = False

    def __str__(self) -> str:
        return 'raw:' + self.y_raw.__str__()

    def _return_value(self, value):
        """
        Return a single value or the value array.

        Depending on property `_single_sample`.

        :param value: return value

        :return:
        """

        if self._single_sample:
            # When a single sample has been selected, the list of values has
            # always length 1.
            return value[0]
        else:
            return value

        pass

    @property
    def y(self):
        y = np.array(self.y_raw)
        yf = y.astype(float)
        if self.y_hole is not None:
            yf[y == self.y_hole] = float('nan')

        return self._return_value(
            ((yf - self.y_reference) * self.y_increment) + self.y_origin)

    def __getitem__(self: T, index) -> T:
        """
        Return the object containing the selected samples.

        :param index: index of selected samples

        :return: object of this class
        """

        # return a copy of the object
        self_copy = copy.copy(self)

        # select items
        try:
            if isinstance(index, (slice, np.ndarray)):
                self_copy.y_raw = self.y_raw[index]
            elif self.y_raw is not None:
                # ensure that y_raw stays iterable
                self_copy.y_raw = self.y_raw[index:index + 1]
                # remember that only a single sample is selected
                self_copy._single_sample = True
            else:
                raise IndexError
        except IndexError as error:
            raise IndexError from error

        # return the copied object
        return self_copy

    def __iter__(self):
        return (float('nan') if y == self.y_hole else (
                (y - self.y_reference) * self.y_increment + self.y_origin)
                for i, y in enumerate(self.y_raw))

    def __len__(self):
        return len(self.y_raw)

    def count(self):
        return len(self.y_raw)


class TraceYT(TraceY):
    """"Y-T trace object"""

    def __init__(self):
        super(TraceYT, self).__init__()
        # 1 / sample rate
        self.x_increment = 0
        # trigger time in seconds
        self.x_origin = 0
        # pre-trigger time in samples
        self.x_reference = 0

        # remember when numpy advanced indexing has been used
        self._numpy_indexing_warn = False

    @property
    def x(self):
        """Return the time axis"""

        if self._numpy_indexing_warn:
            warn('With advanced indexing only the first time index is '
                 'guaranteed to be correct.')
            self._numpy_indexing_warn = False

        return self._return_value(
            ((np.arange(len(self.y_raw)) - self.x_reference)
             * self.x_increment) + self.x_origin)

    @property
    def t(self):
        """Return the time axis"""
        return self.x

    def __iter__(self):
        return ((((i - self.x_reference) * self.x_increment) + self.x_origin,
                 float('nan') if y == self.y_hole
                 else ((y - self.y_reference) * self.y_increment)
                 + self.y_origin)
                for i, y in enumerate(self.y_raw))

    def __getitem__(self: T, index) -> T:
        """
        Return the object containing the selected samples.

        :param index: index of selected samples

        :return: object of this class
        """

        # select samples
        self_copy = super().__getitem__(index)

        # adjust time axis
        if type(index) is slice:
            x_i = index.start or 0
        elif type(index) is np.ndarray:
            if len(self_copy) > 0:
                self_copy._numpy_indexing_warn = True
                if index.dtype == np.bool:
                    x_i = np.nonzero(index)[0][0]
                else:
                    x_i = index[0]
            else:
                x_i = np.nan
        else:
            x_i = index
        # noinspection PyUnresolvedReferences
        self_copy.x_reference -= x_i

        return self_copy

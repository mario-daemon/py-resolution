import math
import re
from enum import Enum
from typing import Protocol, Self

from resolution.utils import round_to_nearest_multiple, rounded

_PATTERN = re.compile(r"^[0-9]+x[0-9]+$")
_STANDARD_16_9_NOMENCLATURE = {
    "144p": (256, 144),
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320)
}

class HasSize(Protocol):
    @property
    def size(self) -> tuple[int, int]: ...

class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...



class Orientation(Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"



class Resolution(tuple):
    """
    A tuple-like class for resolutions. Usage: `Resolution(1280, 720)`.

    Given dimensions must be integers and greater than 0.
    """
    __slots__ = ()
    __match_args__ = ("width", "height")


    def __new__(cls, width: int, height: int) -> Self:

        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(f"width and height must be int; got {type(width)} and {type(height)}")

        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0; got {width}, {height}")

        return super().__new__(cls, (width, height))


    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        Create a Resolution from a string.

        Accepts a WIDTHxHEIGHT pattern, for instance "100x50" or a
        standard 16:9 nomenclature like "1080p", "4K", "2k" etc.

        Full list:

        "144p" (256, 144),
        "240p" (426, 240),
        "360p" (640, 360),
        "480p" (854, 480),
        "720p" (1280, 720),
        "1080p" (1920, 1080),
        "1440p" (2560, 1440),
        "2k" (2560, 1440),
        "4k" (3840, 2160),
        "8k" (7680, 4320)
        """

        if not isinstance(string, str):
            raise TypeError(f"parameter string must be of type str; got {type(string)}")

        string = string.strip().lower()

        if string in _STANDARD_16_9_NOMENCLATURE:
            w, h = _STANDARD_16_9_NOMENCLATURE[string]
            return cls(w, h)

        if len(string) < 3:
            raise ValueError(f"invalid string for Resolution; got '{string}'")

        if not _PATTERN.fullmatch(string):
            raise ValueError(f"invalid string for Resolution; got '{string}'")

        w, h = string.split("x")
        return cls(int(w), int(h))


    @classmethod
    def from_image(cls, img: HasSize) -> Self:
        """
        Create a Resolution from an image.

        The passed object `img` must have a `.size` attribute that returns a tuple as (width, height). Like a `PIL.Image`.
        """

        if not hasattr(img, "size"):
            raise TypeError(f"{type(img).__name__} has no 'size' attribute; expected an object exposing size as (width, height).")

        w, h = img.size

        return cls(w, h)


    @classmethod
    def from_array(cls, array: HasShape) -> Self:
        """
        Create a Resolution from a numpy-style array.

        The passed object `array` must have a `.shape` attribute that returns a tuple as (height, width, ...). Like an OpenCV frame or a Scikit-image image.
        """

        if not hasattr(array, "shape"):
            raise TypeError(f"{type(array).__name__} has no 'shape' attribute; expected an object exposing shape as (height, width, ...).")

        shape: tuple = array.shape

        if len(shape) < 2:
            raise ValueError(f"array must have at least 2 dimensions; got shape {shape}")

        h, w = shape[0], shape[1]

        return cls(w, h)


    @property
    def width(self) -> int:
        return self[0]


    @property
    def height(self) -> int:
        return self[1]


    @property
    def pixels(self) -> int:
        return self.width * self.height


    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / min(self.width, self.height)


    @property
    def simple_ratio(self) -> tuple[int, int]:
        """Aspect ratio returned as a tuple. For instance `(16, 9)` instead of `1.7778`"""
        g = math.gcd(self.width, self.height)
        return (self.width // g, self.height // g)


    @property
    def orientation(self) -> Orientation:
        if self.width > self.height:
            return Orientation.LANDSCAPE
        elif self.height > self.width:
            return Orientation.PORTRAIT
        else:
            return Orientation.SQUARE


    def flipped(self) -> Self:
        """Return a copy with flipped dimensions."""
        return type(self).__call__(self.height, self.width)


    def with_orientation(self, orient: Orientation) -> Self:
        """Return a copy flipped to a specified orientation. No-op for squares."""

        if orient in (self.orientation, Orientation.SQUARE):
            return type(self).__call__(self.width, self.height)

        return self.flipped()


    def to_shape(self, channels: int | None = None) -> tuple[int, ...]:
        """Return a tuple with a numpy/cv2 shape order (height, width), with an optional third element `channels`."""

        if channels is None:
            return (self.height, self.width)

        return (self.height, self.width, channels)


    def fits_inside(self, other_res: Self) -> bool:
        """
        Geometric comparison to check if current resolution fits inside the bounds of `other_res`.
        Equal dimensions also count as fitting.
        """
        return (
            self.width <= other_res.width
            and
            self.height <= other_res.height
        )


    def covers(self, other_res: Self) -> bool:
        """
        Geometric comparison to check if current resolution covers over the bounds of `other_res`.
        Equal dimensions also count as covering.
        """
        return (
            self.width >= other_res.width
            and
            self.height >= other_res.height
        )


    def scaled_by(self, multiplier: float, round_to: int = 1) -> Self:
        """
        Return a copy with dimensions scaled by a factor of `multiplier`.

        Dimensions are rounded to `round_to` value (with round-half-up).
        """

        if multiplier <= 0:
            raise ValueError(f"multiplier must be greater than 0; got {multiplier}")

        if round_to:
            new_w = int(round_to_nearest_multiple(self.width * multiplier, round_to))
            new_h = int(round_to_nearest_multiple(self.height * multiplier, round_to))
        else:
            new_w = rounded(self.width * multiplier)
            new_h = rounded(self.height * multiplier)

        try:
            return type(self).__call__(new_w, new_h)
        except ValueError as e:
            raise ValueError(f"scaled_by({multiplier}, {round_to}) produced an invalid resolution") from e


    def get_fit_scale(self, other_res: Self) -> float:
        """
        Return the scale factor needed to fit entirely inside `other_res`
        with the smallest amount of unused space, preserving aspect ratio.
        """
        return min(other_res.width / self.width, other_res.height / self.height)


    def get_cover_scale(self, other_res: Self) -> float:
        """
        Return the scale factor needed to fully cover `other_res`
        with the smallest amount of overflow, preserving aspect ratio.
        """
        return max(other_res.width / self.width, other_res.height / self.height)


    def scaled_to_fit(self, other_res: Self) -> Self:
        """
        Return a copy of current resolution scaled to fit within `other_res`, preserves aspect ratio.

        Internally, uses `self.get_fit_scale()` and `self.scaled_by()`.
        """
        return self.scaled_by(self.get_fit_scale(other_res))


    def scaled_to_cover(self, other_res: Self) -> Self:
        """
        Return a copy of current reoslution scaled to cover `other_res`, preserves aspect ratio.

        Internally, uses `self.get_cover_scale()` and `self.scaled_by()`.
        """
        return self.scaled_by(self.get_cover_scale(other_res))


    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.width}, {self.height})"


    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


    def __lt__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self.pixels < other.pixels


    def __gt__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self.pixels > other.pixels


    def __le__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self.pixels <= other.pixels


    def __ge__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self.pixels >= other.pixels


    def __reduce__(self):
        return (type(self), (self.width, self.height))


    def __mul__(self, other) -> Self:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.scaled_by(float(other))

    __rmul__ = __mul__

    def __truediv__(self, other) -> Self:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.scaled_by(1 / float(other))

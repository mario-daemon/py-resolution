# Py-Resolution

A simple datatype for image resolutions, scaling, aspect ratios, orientations etc. With support for your favourite libraries like Pillow, NumPy, OpenCV, Scikit-image etc.

## Usage

### Basic

```python
from resolution import Resolution as res

r = res(1920, 1080)
print(r)               # 1920x1080
print(r.width)         # 1920
print(r.height)        # 1080
print(r.pixels)        # 2073600
print(r.aspect_ratio)  # 1.77777...
print(r.simple_ratio)  # (16, 9)
print(r.orientation)   # Orientation.LANDSCAPE

# Unpack like a normal tuple
w, h = res
```

### Alternative Construction

```python
from resolution import Resolution as res

# From standard 16:9 nomenclature strings or "WxH" format
r1 = res.from_str("1080p")      # Resolution(1920, 1080)
r2 = res.from_str("4k")         # Resolution(3840, 2160)
r3 = res.from_str("1280x720")   # Resolution(1280, 720)

# PIL / Pillow Images (reads .size)
from PIL import Image
img = Image.new("RGB", (800, 600))
r_pil = res.from_image(img)     # Resolution(800, 600)

# Numpy, OpenCV, or Scikit-Image arrays (reads .shape)
import numpy as np
array = np.zeros((1080, 1920, 3), dtype=np.uint8)
r_arr = res.from_array(array)   # Resolution(1920, 1080)
```


### Orientations and flipping

```python
from resolution import Resolution as res, Orientation as orient

r = res(1920, 1080)

# Flip width and height
flipped = r.flipped()
print(flipped)  # 1080x1920

# Force a specific orientation
portrait_res = r.with_orientation(orient.PORTRAIT)
print(portrait_res)  # 1080x1920

# Orientation check
print(r.orientation == orient.LANDSCAPE)  # True
```


### Scaling Logic

```python
from resolution import Resolution as res

r = res(1920, 1080)

# Scale by a multiplier (supports rounding)
scaled = r.scaled_by(0.5)
rounded_scaled = r.scaled_by(0.33, round_to=16)  # Rounds dimensions to nearest multiple of 16
print(scaled)                           # 960x540
print(rounded_scaled)                   # 640x352

# Operator shorthand
double_res = r * 2                      # 3840x2160
half_res = r / 2                        # 960x540

# Scale to fit inside or cover target bounds (aspect-ratio preserved)
bounds = res(1000, 1000)

fit_res = r.scaled_to_fit(bounds)       # Fits inside target bounds
cover_res = r.scaled_to_cover(bounds)   # Covers target bounds completely
print(fit_res)                          # 1000x563
print(cover_res)                        # 1778x1000

# Retrieve raw scale factors
scale_fit = r.get_fit_scale(bounds)
scale_cover = r.get_cover_scale(bounds)
print(scale_fit)                        # 0.520833...
print(scale_cover)                      # 0.925925...
```


### Spatial & Volumetric Comparisons

```python
from resolution import Resolution as res

r1 = res(1280, 720)
r2 = res(1920, 1080)

# Geometric containment checks
print(r1.fits_inside(r2))  # True
print(r2.covers(r1))       # True

# Volumetric comparisons based on total pixel count
print(r1 < r2)   # True (921,600 pixels < 2,073,600 pixels)
print(r1 >= r2)  # False

# Pure equality compares dimensions instead
r2 = res(1280, 720)
print(r1 == r2)  # True

r2 = res(720, 1280)
print(r1 == r2)  # False
```

### Framework & Interop Helpers

```python
from resolution import Resolution as res

r = res(1920, 1080)

# Convert to Numpy/OpenCV shape tuple (height, width[, channels])
shape_2d = r.to_shape()     # (1080, 1920)
shape_3d = r.to_shape(3)    # (1080, 1920, 3)

# Pattern matching (Python 3.10+)
match r:
    case res(1920, 1080):
        print("Full HD detected")
```

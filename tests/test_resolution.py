import pickle

import cv2
import numpy as np
import pytest
import skimage
from PIL import Image
from pytest import fixture, mark

from resolution.core import Orientation, Resolution

parametrize = mark.parametrize

@fixture
def fake_image_path(tmp_path) -> str:

    img = Image.new("RGB", (100, 50))
    path = tmp_path / "fake.png"
    img.save(path)

    return str(path)


class TestCreation:


    @parametrize("w,h", [
        (100, 50), (1, 1), (20, 30)
    ])
    def test_normal(self, w: int , h: int):

        _ = Resolution(w, h)


    @parametrize("bad_w,bad_h", [(0, 0), (-10, -10), (True, False)])
    def test_rejects_invalid_numbers(self, bad_w: int, bad_h: int):

        with pytest.raises(ValueError, match="must be > 0"):
            _ = Resolution(bad_w, bad_h)


    @parametrize("bad_w,bad_h", [
        ("500", 100), (1.6, 4), (None, None), (range(3), [1,2,3])
    ])
    def test_rejects_invalid_types(self, bad_w, bad_h):

        with pytest.raises(TypeError, match="must be int"):
            _ = Resolution(bad_w, bad_h)


    @parametrize("string,expected_w,expected_h", [
        ("1920x1080", 1920, 1080),
        ("1x1", 1, 1),
        ("512x128", 512, 128),
        ("06x06", 6, 6),
        ("720P", 1280, 720),
        ("4k", 3840, 2160)
    ])
    def test_from_str(self, string: str, expected_w, expected_h):

        r = Resolution.from_str(string)

        assert type(r.width) is int and type(r.height) is int

        assert r.width == expected_w
        assert r.height == expected_h


    @parametrize("bad_string", [
        ("abcxdef"), ("123x"), ("x123"), ("100x100x100"), ("0432"), ("hellOwoRlD"), ("-1080x1920")
    ])
    def test_from_str_rejects_invalid_strings(self, bad_string: str):

        with pytest.raises(ValueError, match="invalid string"):
            _ = Resolution.from_str(bad_string)


    @parametrize("bad_string", [
        5, 3.14, True, None, [1,2,3], (6, 7)
    ])
    def test_from_str_rejects_invalid_types(self, bad_string):

        with pytest.raises(TypeError):
            _ = Resolution.from_str(bad_string)


    @parametrize("unpackable", [list, tuple])
    def test_from_unpack(self, unpackable: type):

        data = unpackable([100, 50])

        _ = Resolution(*data)


    def test_from_pillow(self):

        img = Image.new("RGB", (100, 50))

        _ = Resolution.from_image(img)


    @parametrize("bad_img",[
        1, "image", True, None
    ])
    def test_from_pillow_rejects_invalid_types(self, bad_img):

        with pytest.raises(TypeError, match="'size' attribute"):
            _ = Resolution.from_image(bad_img)


    @parametrize("array", [
        np.zeros((50, 100, 3), dtype=np.uint8),
        np.zeros((50, 100, 3), dtype=np.float32)
    ])
    def test_from_array_numpy(self, array: np.ndarray):

        r = Resolution.from_array(array)

        # check that width/height are correct
        # since they come reversed in the array
        assert r.width == 100
        assert r.height == 50


    def test_from_array_opencv(self, fake_image_path):

        img_array = cv2.imread(fake_image_path)
        r = Resolution.from_array(img_array)

        # check that width/height are correct
        # since they come reversed in the array
        assert r.width == 100
        assert r.height == 50


    def test_from_array_scikit(self, fake_image_path):

        img_array = skimage.io.imread(fake_image_path)
        r = Resolution.from_array(img_array)

        # check that width/height are correct
        # since they come reversed in the array
        assert r.width == 100
        assert r.height == 50


    @parametrize("bad_array", [
        [1,2,3], (1,2,3), True, False, None, "array"
    ])
    def test_from_array_rejects_invalid_types(self, bad_array):

        with pytest.raises(TypeError, match="'shape' attribute"):
            _ = Resolution.from_array(bad_array)



class TestOrientation:


    @parametrize("res,expected_orient", [
        (Resolution(100, 50), Orientation.LANDSCAPE),
        (Resolution(50, 100), Orientation.PORTRAIT),
        (Resolution(50, 50), Orientation.SQUARE)
    ])
    def test_orientation(self, res: Resolution, expected_orient: Orientation):

        assert res.orientation == expected_orient


    @parametrize("res,expected_orient", [
        (Resolution(100, 50), Orientation.PORTRAIT),
        (Resolution(50, 100), Orientation.LANDSCAPE),
        (Resolution(50, 50), Orientation.SQUARE)
    ])
    def test_flipped(self, res: Resolution, expected_orient: Orientation):

        assert res.flipped().orientation == expected_orient


    @pytest.mark.parametrize("orient", [
        Orientation.LANDSCAPE, Orientation.PORTRAIT, Orientation.SQUARE
    ])
    def test_with_orientation(self, orient: Orientation):

        r = Resolution(100, 50)

        if orient == Orientation.SQUARE:
            assert r.with_orientation(orient).orientation == r.orientation # no-op
        else:
            assert r.with_orientation(orient).orientation == orient



class TestScaling:


    @parametrize("r,multiplier,expected_w,expected_h", [
        (Resolution(100, 50), 2.0, 200, 100),
        (Resolution(100, 50), 0.5, 50, 25),
        (Resolution(100, 50), 1.0, 100, 50),
    ])
    def test_basic_scaling(self, r: Resolution, multiplier, expected_w, expected_h):

        scaled = r.scaled_by(multiplier)

        assert scaled.width == expected_w
        assert scaled.height == expected_h


    def test_scaling_by_1_is_unchanged(self):

        r = Resolution(100, 50)
        scaled = r.scaled_by(1)

        assert scaled == r


    @parametrize("mult", [(0), (-1)])
    def test_rejects_non_positive_multiplier(self, mult: float):

        r = Resolution(100, 50)

        with pytest.raises(ValueError, match="multiplier must be greater than 0"):
            r.scaled_by(mult)


    def test_raises_value_error_on_invalid_result(self):

        r = Resolution(1, 1)

        with pytest.raises(ValueError, match=r"produced an invalid resolution"):
            r.scaled_by(0.1)


    @parametrize("r,round_to,expected_dims", [
        (Resolution(19, 9), 2, (20, 10)), # half-up
        (Resolution(38, 19), 6, (36, 18)),
        (Resolution(38, 78), 4, (40, 80)), # half-up
        (Resolution(69, 58), 7, (70, 56)),
        (Resolution(12, 20), 8, (16, 24)) # half-up
    ])
    def test_rounding(self, r: Resolution, round_to: int, expected_dims: tuple[int, int]):

        scaled = r.scaled_by(1, round_to)

        assert scaled == expected_dims


    @parametrize("round_to,expected_w,expected_h", [
        # 100*1.3=130, 50*1.3=65
        (16, 128, 64),
        (8, 128, 64),
        (4, 132, 64),
        (2, 130, 66),
    ])
    def test_round_to_multiple(self, round_to: int, expected_w: int, expected_h: int):

        r = Resolution(100, 50)

        scaled = r.scaled_by(1.3, round_to=round_to)

        assert scaled.width == expected_w
        assert scaled.height == expected_h
        assert scaled.width % round_to == 0
        assert scaled.height % round_to == 0


    def test_get_fit(self):

        r = Resolution(100, 50)
        bounds = Resolution(20, 20)
        scale = r.get_fit_scale(bounds)

        assert scale == 0.2 # 100 * 0.2 = 20


    def test_get_cover(self):

        r = Resolution(100, 50)
        bounds = Resolution(20, 20)
        scale = r.get_cover_scale(bounds)

        assert scale == 0.4 # 50 * 0.4 = 20


    def test_get_any_scale_coincide_given_the_same_aspect_ratio(self):

        r = Resolution(100, 50)
        bounds = Resolution(200, 100)

        assert r.get_fit_scale(bounds) == 2.0
        assert r.get_cover_scale(bounds) == 2.0



class TestComparison:


    def test_geometric_equality(self):

        r1, r2 = Resolution(100, 50), Resolution(100, 50)

        comparison = r1.width == r2.width and r1.height == r2.height # both dims match

        assert (r1 == r2) == comparison


    def test_geometric_inequality(self):

        r1, r2 = Resolution(100, 50), Resolution(50, 100)

        comparison = r1.width != r2.width or r1.height != r2.height # any dims mismatch

        assert (r1 != r2) == comparison


    def test_fitting(self):

        r1 = Resolution(100, 50)
        r2 = Resolution(101, 50)

        comparison = r1.width <= r2.width and r1.height <= r2.height # both dims are smaller or equal

        assert r1.fits_inside(r2) == comparison


    def test_covering(self):

        r1 = Resolution(100, 50)
        r2 = Resolution(99, 49)

        comparison = r1.width >= r2.width and r1.height >= r2.height # both dims are larger or equal

        assert r1.covers(r2) == comparison


    @parametrize("r1", [
        Resolution(99, 50),
        Resolution(100, 49),
        Resolution(99, 49)
    ])
    def test_fitting_exclusively(self, r1):

        r2 = Resolution(100, 50)

        assert (r1.fits_inside(r2) and not r1.covers(r2)) == True # check that r1 fits within by strictly being smaller, and not equal


    @parametrize("r1", [
        Resolution(101, 50),
        Resolution(100, 51),
        Resolution(101, 51)
    ])
    def test_covering_exclusively(self, r1):

        r2 = Resolution(100, 50)

        assert (r1.covers(r2) and not r1.fits_inside(r2)) == True # check that r1 covers over by strictly being larger, and not equal


    def test_both_fitting_and_covering_means_equality(self):

        r1 = Resolution(100, 50)
        r2 = Resolution(100, 50)

        assert ( r1.covers(r2) and r1.fits_inside(r2) ) == ( r1 == r2 )


    @parametrize("r1,r2", [
        (Resolution(101, 50), Resolution(50, 100)), # r1 is greater
        (Resolution(100, 50), Resolution(50, 100)) # equal volume
    ])
    def test_volumetric_comparison_and_equality(self, r1: Resolution, r2: Resolution):

        comparison = r1.pixels >= r2.pixels

        assert (r1 >= r2) == comparison



class TestMisc:


    def test_normal_access_syntax(self):

        r = Resolution(100, 50)

        assert r.width == 100
        assert r.height == 50


    def test_getattribute_dunder(self):

        r = Resolution(100, 50)

        assert r.__getattribute__("width") == 100
        assert r.__getattribute__("height") == 50


    def test_getitem_dunder(self):

        r = Resolution(100, 50)

        assert r.__getitem__(0) == 100
        assert r.__getitem__(1) == 50


    def test_iteration(self):

        r = Resolution(100, 50)

        for i in r:
            pass

        assert r.__iter__() is not None


    def test_repr_is_evaluable(self):

        r = Resolution(100, 50)
        recreated = eval(r.__repr__())

        assert recreated == r


    def test_to_shape(self):

        r = Resolution(100, 50)

        shape = r.to_shape()
        assert shape == (50, 100)

        arr = np.ndarray(shape)
        assert arr.shape == (50, 100)

        shape = r.to_shape(3)
        assert shape == (50, 100, 3)

        arr = np.ndarray(shape)
        assert arr.shape == (50, 100, 3)


    def test_to_shape_opencv(self, fake_image_path):
        r = Resolution(100, 50)

        img_array = cv2.imread(fake_image_path)
        assert img_array.shape == r.to_shape(3)

        blank = np.zeros(r.to_shape(3), dtype=np.uint8)
        assert Resolution.from_array(blank) == r


    def test_to_shape_scikit(self, fake_image_path):
        r = Resolution(100, 50)

        img_array = skimage.io.imread(fake_image_path)
        assert img_array.shape == r.to_shape(3)

        blank = np.zeros(r.to_shape(3), dtype=np.uint8)
        assert Resolution.from_array(blank) == r


    @parametrize("r,aspect", [
        (Resolution(1920, 1080), (16, 9)),
        (Resolution(300, 140), (15, 7)),
        (Resolution(640, 480), (4, 3))
    ])
    def test_simple_aspect(self, r: Resolution, aspect: tuple[int, int]):

        assert r.simple_ratio == aspect


    def test_multiplication(self):

        r = Resolution(100, 50)

        assert r * 2 == (200, 100)
        assert 2 * r == (200, 100)


    def test_division(self):

        r = Resolution(100, 50)

        assert r / 2 == (50, 25)


    def test_serialization(self):

        r = Resolution(100, 50)

        data = pickle.dumps(r)
        restored = pickle.loads(data)

        assert isinstance(restored, Resolution)
        assert restored == r


    def test_match_case(self):

        r = Resolution(100, 50)

        match r:
            case Resolution(100, 50):
                pass
            case (100, 50):
                pass

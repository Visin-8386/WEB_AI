import numpy as np
from src.data import preprocessing

def test_resize_and_pad():
    img = np.ones((10, 20), dtype=np.uint8) * 255
    out = preprocessing.resize_and_pad(img, size=28, padding=4)
    assert out.shape == (28, 28)
    print('test_resize_and_pad passed')

def test_increase_stroke_thickness():
    img = np.zeros((28,28), dtype=np.uint8)
    img[10:18, 10:18] = 255
    out = preprocessing.increase_stroke_thickness(img)
    assert out.shape[0] > 28 and out.shape[1] > 28
    print('test_increase_stroke_thickness passed')

def test_group_boxes():
    boxes = [(10, 10, 5, 5), (12, 12, 5, 5), (50, 50, 5, 5)]
    sorted_boxes = preprocessing.group_boxes(boxes, row_threshold=10)
    assert isinstance(sorted_boxes, list)
    print('test_group_boxes passed')

def test_apply_gaussian_blur():
    img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    blurred = preprocessing.apply_gaussian_blur(img)
    assert blurred.shape == img.shape
    print('test_apply_gaussian_blur passed')

def test_enhance_black_background():
    img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    enhanced = preprocessing.enhance_black_background(img)
    assert enhanced.shape == img.shape
    print('test_enhance_black_background passed')

def run_all_tests():
    test_resize_and_pad()
    test_increase_stroke_thickness()
    test_group_boxes()
    test_apply_gaussian_blur()
    test_enhance_black_background()
    print('Tất cả các test đã vượt qua!')

if __name__ == '__main__':
    run_all_tests()

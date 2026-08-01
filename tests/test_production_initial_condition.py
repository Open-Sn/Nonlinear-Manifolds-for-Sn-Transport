import numpy as np

from Transport_Driver_Benchmark_1D import make_production_initial_condition


def test_characterization_of_preserved_production_sigmoid():
    """Characterize current code behavior; this is not paper validation."""
    coordinates = np.array([0.0, 0.05, 0.1, 0.2, 1.0])
    initial = make_production_initial_condition(coordinates)
    angular_blocks = initial.reshape(4, coordinates.size)
    expected_profile = 1 - 1 / (
        1 + np.exp(-100 * (coordinates - 0.1))
    )

    np.testing.assert_array_equal(angular_blocks[:3], np.zeros_like(angular_blocks[:3]))
    np.testing.assert_allclose(angular_blocks[3], expected_profile, rtol=0.0, atol=0.0)
    assert angular_blocks[3, 0] > 0.99
    assert angular_blocks[3, 2] == 0.5
    assert angular_blocks[3, 3] < 5.0e-5
    assert angular_blocks[3, 0] > angular_blocks[3, -1]

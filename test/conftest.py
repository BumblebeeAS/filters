"""Pytest fixtures shared across the filters/test/ suite."""

import pytest
import rclpy


@pytest.fixture(scope="module")
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()

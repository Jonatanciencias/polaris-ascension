from __future__ import annotations

import pytest

import src.benchmarking.production_kernel_benchmark as pkb


class _FakeDevice:
    def __init__(
        self,
        *,
        name: str,
        vendor: str = "AMD",
        version: str = "OpenCL 3.0",
        driver_version: str = "1.0",
    ) -> None:
        self.name = name
        self.vendor = vendor
        self.version = version
        self.driver_version = driver_version


class _FakePlatform:
    def __init__(
        self,
        *,
        name: str,
        vendor: str = "Mesa",
        version: str = "OpenCL",
        devices: list[_FakeDevice] | None = None,
    ) -> None:
        self.name = name
        self.vendor = vendor
        self.version = version
        self._devices = devices or []

    def get_devices(self, device_type=None):  # noqa: ANN001
        return list(self._devices)


def _sample_platforms() -> list[_FakePlatform]:
    return [
        _FakePlatform(
            name="Clover",
            vendor="Mesa",
            version="OpenCL 1.1",
            devices=[_FakeDevice(name="AMD Radeon RX 590", vendor="AMD", version="OpenCL 1.1")],
        ),
        _FakePlatform(
            name="rusticl",
            vendor="Mesa/X.org",
            version="OpenCL 3.0",
            devices=[_FakeDevice(name="AMD Radeon RX 590", vendor="AMD", version="OpenCL 3.0")],
        ),
    ]


def test_opencl_selection_defaults_to_available_amd_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pkb.ENV_OPENCL_PLATFORM, raising=False)
    monkeypatch.delenv(pkb.ENV_OPENCL_DEVICE, raising=False)
    monkeypatch.setattr(pkb.cl, "get_platforms", lambda: _sample_platforms())

    platform, device, info = pkb._select_opencl_runtime(
        opencl_platform=None,
        opencl_device=None,
    )

    assert platform.name == "Clover"
    assert "AMD Radeon RX 590" in device.name
    assert info["platform_selector"] == "auto"
    assert info["device_selector"] == "auto"


def test_opencl_selection_honors_explicit_platform_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pkb.cl, "get_platforms", lambda: _sample_platforms())

    platform, _, info = pkb._select_opencl_runtime(
        opencl_platform="rusticl",
        opencl_device=None,
    )

    assert platform.name == "rusticl"
    assert info["platform_selector"] == "rusticl"


def test_opencl_selection_honors_env_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pkb.ENV_OPENCL_PLATFORM, "rusticl")
    monkeypatch.setenv(pkb.ENV_OPENCL_DEVICE, "rx 590")
    monkeypatch.setattr(pkb.cl, "get_platforms", lambda: _sample_platforms())

    platform, device, info = pkb._select_opencl_runtime(
        opencl_platform=None,
        opencl_device=None,
    )

    assert platform.name == "rusticl"
    assert "RX 590" in device.name
    assert info["platform_selector_from_env"] is True
    assert info["device_selector_from_env"] is True


def test_opencl_selection_raises_clear_error_for_unknown_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pkb.cl, "get_platforms", lambda: _sample_platforms())

    with pytest.raises(ValueError, match="matched none"):
        pkb._select_opencl_runtime(
            opencl_platform="nonexistent",
            opencl_device=None,
        )

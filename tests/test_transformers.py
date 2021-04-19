import numpy as np

try: # only test if xarray present

    import xarray as xr
    from models import NoOpTransformation

    def test_noop_transformer():
        ones = np.ones((4,4), dtype=np.float32)
        red = xr.DataArray(1*ones, dims=("y", "x"), name=f"red")
        nir = xr.DataArray(2*ones, dims=("y", "x"), name=f"nir")

        ds = xr.Dataset(data_vars={"red": red, "nir": nir})

        transform = NoOpTransformation()

        result = transform.compute(ds)

        assert 'x' in result.dims
        assert 'y' in result.dims
        assert 'variable' in result.dims

except ImportError:
    pass

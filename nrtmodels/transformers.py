import numpy as np

from .base import Model, NoOp

try: # only enable if xarray present

    import xarray as xr

    class Transformation: # Mock
        
        def __init__(self):
            pass

        def compute(self, data: xr.Dataset) -> xr.Dataset:
            raise NotImplementedError

    
    class NoOpTransformation(Transformation):

        def __init__(self):
            self.model = NoOp()

        def compute(self, ds: xr.Dataset) -> xr.Dataset:
            squashed = ds.to_array().transpose("y", "x", "variable")
            data = squashed.data.astype(np.float32) / 10000.

            result = self.model.predict(data)

            da = xr.DataArray(result, dims=("y", "x", "variable"), name="result")
            return xr.Dataset(data_vars={"result": da})

except ImportError:
    pass

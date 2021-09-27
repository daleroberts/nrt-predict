
import numpy as np
import joblib

from .base import Model
from pathlib import Path

BANDNAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def cosine(X, Y):
    """
    Cosine distance between `X` and `Y` calculated over axis=2.
    """
    nX = 1 / np.sqrt(np.sum(np.square(X), axis=2))
    nY = 1 / np.sqrt(np.sum(np.square(Y), axis=2))
    XX = np.einsum("ij,ijk->ijk", nX, X)
    YY = np.einsum("ij,ijk->ijk", nY, Y)
    return 1.0 - np.einsum("ijk,ijk->ij", XX, YY)


def euclidean(X, Y):
    """
    Euclidean distance between `X` and `Y` calculated over axis=2.
    """
    return np.sqrt(np.sum(np.square(X - Y), axis=2))


def braycurtis(X, Y):
    """
    Bray-Curtis distance between `X` and `Y` calculated over axis=2.
    """
    return np.sum(np.absolute(X - Y), axis=2) / np.sum(np.absolute(X + Y), axis=2)


def nldr(X, Y, i=0, j=1):
    """
    Normalised band difference ratio feature between `X` and `Y` calculated over axis=2.
    """
    XA, XB = X[:, :, i], X[:, :, j]
    YA, YB = Y[:, :, i], Y[:, :, j]
    numerX = 2 * (XA ** 2 + XB ** 2) + 1.5 * XB + 0.5 * XA
    denomX = XA + XB + 0.5
    numerY = 2 * (YA ** 2 + YB ** 2) + 1.5 * YB + 0.5 * YA
    denomY = YA + YB + 0.5
    return np.absolute(numerX / denomX - numerY / denomY)


def features(obs, ftrexprs, ref=None, bandnames=BANDNAMES):
    """
    Generate model features based on observation (`obs`), reference (`ref`),
    and feature expressions (`ftrexprs`). The band names corresponding to `obs`
    may be given.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bandnos = {b: i for i, b in enumerate(bandnames)}
        ffexprs = []
        for f in ftrexprs:
            if f.startswith("nldr"):
                for b in bandnames:
                    f = f.replace(b, str(bandnos[b]))
            ffexprs.append(f)
        env = {
            **{b: obs[:, :, bandnos[b]] for b in bandnames},
            **{s: getattr(np, s) for s in dir(np)},
            **{
                "cosine": cosine,
                "euclidean": euclidean,
                "braycurtis": braycurtis,
                "nldr": nldr,
                "ref": ref,
                "obs": obs,
            },
        }
        return np.stack([eval(e, {"__builtins__": {}}, env) for e in ffexprs], axis=-1)


class CloudAndShadowDetect(Model):

    """
    Cloud and shadow detection model.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = Path(__file__).resolve().parent
        self.required_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        self._model1, self._exprs1 = joblib.load(path / "cloud-shadow-1.pkl.xz")
        self._model2, self._exprs2 = joblib.load(path / "cloud-spectral-1.pkl.xz")

    def generate_features1(self, ref, obs):
        ftrs = features(obs, self._exprs1, ref)
        return ftrs.reshape((-1, ftrs.shape[-1]))

    def generate_features2(self, ref, obs):
        ftrs = features(obs, self._exprs2, ref)
        return ftrs.reshape((-1, ftrs.shape[-1]))

    def predict(self, mask, ref, obs):
        bad = np.logical_or(~np.isfinite(ref), ~np.isfinite(obs)).any(axis=2)
        ref[bad] = np.nan
        obs[bad] = np.nan

        X = self.generate_features1(ref, obs)
        good = np.isfinite(X).all(axis=1)
        X[~good] = 0
        yhat1 = self._model1.predict_proba(X).astype(np.float32).reshape((obs.shape[0], obs.shape[1], -1))

        X = self.generate_features2(ref, obs)
        good = np.isfinite(X).all(axis=1)
        X[~good] = 0
        yhat2 = self._model2.predict_proba(X).astype(np.float32).reshape((obs.shape[0], obs.shape[1], -1))

        yhat = 0.5 * (yhat1 + yhat2)

        cloud = yhat[:, :, 1] > 0.5
        shadow = yhat[:, :, 2] > 0.5

        from skimage import morphology

        cloud = morphology.binary_erosion(cloud, morphology.disk(6))
        cloud = morphology.binary_dilation(cloud, morphology.diamond(20))

        shadow = morphology.binary_erosion(shadow, morphology.disk(6))
        shadow = morphology.binary_dilation(shadow, morphology.diamond(20))

        result = np.zeros((obs.shape[0], obs.shape[1]), dtype=np.int8)
        result[cloud] = 1
        result[shadow] = 2

        return result


class UnsupervisedVegetation1(Model):
    """
    Unsupervised Vegetation Change.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    required_bands = ["B04", "B05", "B11", "B12"]

    def generate_features(self, ref, obs):
        def diff(expr):
            return self.eval_expr(expr, obs) - self.eval_expr(expr, ref) 
        exprs = ["B05*log(1+B12)",
                 "(B04-B11)/(B04+B11)", 
                 "(B05-B11)/(B05+B11)"]
        return np.dstack([diff("-1*B05*log(1+B12)")] + [self.eval_expr(expr, obs) for expr in exprs])

    def predict(self, mask, pre, pst):
        from sklearn import semi_supervised
        from skimage import morphology

        F = self.generate_features(pre, pst)

        self.log(f"Features shape: {F.shape}")

        good = np.isfinite(F).all(axis=2)
        F[~good] = 0

        cutoff = np.nanpercentile(F[:,:,0], 99, axis=(0,1))

        pos = F[:,:,0] > cutoff

        lbls = np.zeros((F.shape[0], F.shape[1]), dtype=np.int8)
        
        pos = morphology.binary_erosion(pos, morphology.diamond(3))
        unknown = morphology.binary_dilation(pos, morphology.diamond(20))
        focus = morphology.binary_dilation(pos, morphology.diamond(50))

        outliers = np.zeros((F.shape[0], F.shape[1]), dtype=np.int8)
        outliers[unknown] = -1
        outliers[pos] = 1

        X = F[focus].reshape((-1, F.shape[-1]))
        y = outliers[focus].reshape((-1,))

        lblspread = semi_supervised.LabelSpreading(kernel="knn", alpha=0.1, max_iter=100, n_neighbors=30, n_jobs=-1)
        lblspread.fit(X, y)

        outliers[focus] = lblspread.transduction_
        outliers = outliers.reshape(mask.shape)

        outliers[mask == 0] = 0

        outliers = morphology.binary_closing(outliers, morphology.diamond(5))

        lbls[outliers == 1] = 1

        neg = F[:,:,0] < -1 * cutoff
        neg = morphology.binary_erosion(neg, morphology.diamond(3))
        unknown = morphology.binary_dilation(neg, morphology.diamond(20))
        focus = morphology.binary_dilation(neg, morphology.diamond(50))

        outliers = np.zeros((F.shape[0], F.shape[1]), dtype=np.int8)
        outliers[unknown] = -1
        outliers[neg] = 1

        X = F[focus].reshape((-1, F.shape[-1]))
        y = outliers[focus].reshape((-1,))

        lblspread = semi_supervised.LabelSpreading(kernel="knn", alpha=0.1, max_iter=100, n_neighbors=30, n_jobs=-1)
        lblspread.fit(X, y)

        outliers[focus] = lblspread.transduction_
        outliers = outliers.reshape(mask.shape)

        outliers[mask == 0] = 0

        outliers = morphology.binary_closing(outliers, morphology.diamond(5))

        lbls[outliers == 1] = 2

        return lbls[:, :, np.newaxis]

class SupervisedVegetationChange1(Model):
    """
    Supervised Vegetation Change 1.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = Path(__file__).resolve().parent
        self._model = joblib.load(path / 'veg-change-1.pkl.xz')

    def generate_features(self, ref, obs):
        def diff(expr):
            return self.eval_expr(expr, obs) - self.eval_expr(expr, ref) 
        exprs = ["B05*log(1+B12)", "(B04-B11)/(B04+B11)", "(B05-B11)/(B05+B11)"]
        return np.dstack([diff("-1*B05*log(1+B12)")] + [self.eval_expr(expr, obs) for expr in exprs])

    def predict(self, mask, ref, obs):
        from sklearn import cluster, mixture, linear_model
        from skimage import morphology

        F = self.generate_features(ref, obs)

        self.log(f"Features shape: {F.shape}")

        good = np.isfinite(F).all(axis=2)
        good[mask == 0] = False
        F[~good] = 0
        X = F.reshape((-1, F.shape[-1]))

        model = self._model

        lbls = model.predict(X).astype(np.int8)
        lbls = lbls.reshape((obs.shape[0], obs.shape[1]))

        lbl1 = morphology.binary_closing(lbls == 1, morphology.diamond(3))
        lbl2 = morphology.binary_closing(lbls == 2, morphology.diamond(3))
        water = morphology.binary_dilation(mask==5, morphology.diamond(3))

        lbls[:,:] = 0
        lbls[lbl1] = 1
        lbls[lbl2] = 2
        lbls[water] = 0

        return lbls

class VegetationTernary(Model):
    def predict(self, mask, obs):
        exprs = ["B05*log(1+B12)", "(B04-B11)/(B04+B11)", "(B05-B11)/(B05+B11)"]
        return np.dstack([self.eval_expr(expr, obs) for expr in exprs])

class VegetationChangeTernary(Model):
    def predict(self, mask, ref, obs):
        def diff(expr):
            return self.eval_expr(expr, obs) - self.eval_expr(expr, ref) 
        exprs = ["B05*log(1+B12)", "(B04-B11)/(B04+B11)", "(B05-B11)/(B05+B11)"]
        return np.dstack([diff(expr) for expr in exprs])

class WaterChangeTernary(Model):
    def predict(self, mask, ref, obs):
        exprs = ["(B03 - B8A) / (B03 + B8A)", "(B03 - B08) / (B03 + B08)", "(B03 - B11) / (B03 + B11)"]
        return np.dstack([self.eval_expr(expr, obs) - self.eval_expr(expr, ref) for expr in exprs])


class ExcessWater1(Model):

    """
    Excess water detection model 1.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    required_bands = ["B03", "B08", "B8A", "B11"]

    def generate_features(self, ref, pst):
        exprs = ["(B03 - B8A) / (B03 + B8A)", "(B03 - B08) / (B03 + B08)", "(B03 - B11) / (B03 + B11)"]
        results = [self.eval_expr(exprs[0], pst)]
        results.extend([self.eval_expr(expr, pst) - self.eval_expr(expr, ref) for expr in exprs])
        return np.dstack(results)

    def predict(self, mask, ref, pst):
        nodata = np.logical_or(np.isnan(ref).any(axis=2), np.isnan(pst).any(axis=2))

        X = self.generate_features(ref, pst)
        X = np.prod(1 + np.clip(X, 0, None), axis=-1)
        X[~np.isfinite(X)] = 0
        cloud = np.nanmean(X[mask == 2])
        water = np.nanmean(X[mask == 5])
        cutoff = 0.5 * (cloud + water)
        X -= max(1, cutoff)
        X /= min(1, np.nanmax(X))
        X = np.clip(X, 0, 1, out=X)

        X[nodata] = 0

        return X


class ExcessWater2(Model):

    """
    Excess water detection model 2.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    required_bands = ["B03", "B08", "B8A", "B11"]

    def generate_features(self, ref, pst):
        exprs = ["(B03 - B8A) / (B03 + B8A)", "(B03 - B08) / (B03 + B08)", "(B03 - B11) / (B03 + B11)"]
        results = [self.eval_expr(exprs[0], pst)]
        results.extend([self.eval_expr(expr, pst) - self.eval_expr(expr, ref) for expr in exprs])
        return np.dstack(results)

    def predict(self, mask, ref, pst):
        from sklearn import semi_supervised
        from skimage import morphology

        nodata = np.logical_or(np.isnan(ref).any(axis=2), np.isnan(pst).any(axis=2))

        X = self.generate_features(ref, pst)
        X[nodata] = 0

        outliers = np.prod(1 + np.clip(X, 0, None), axis=-1)
        outliers[~np.isfinite(outliers)] = 0
        cloud = np.nanmean(outliers[mask == 2])
        water = np.nanmean(outliers[mask == 5])
        cutoff = 0.5 * (cloud + water)
        outliers -= max(1, cutoff)
        outliers /= min(1, np.nanmax(outliers))
        outliers = np.clip(outliers, 0, 1, out=outliers)

        known = morphology.binary_erosion(outliers, morphology.diamond(6))
        unknown = morphology.binary_dilation(outliers, morphology.diamond(20))
        focus = morphology.binary_dilation(outliers, morphology.diamond(30))

        outliers[:, :] = 0
        outliers[unknown] = -1
        outliers[known] = 1

        y = outliers[focus].reshape((-1,))
        X = X[focus].reshape((-1, X.shape[-1]))

        lblspread = semi_supervised.LabelSpreading(kernel="knn", alpha=0.8, max_iter=100, n_neighbors=20, n_jobs=1)
        lblspread.fit(X, y)

        self.log(f"Iters: {lblspread.n_iter_}")

        outliers[focus] = lblspread.transduction_
        outliers = outliers.reshape(mask.shape)

        outliers[nodata] = 0

        return outliers[:, :, np.newaxis]

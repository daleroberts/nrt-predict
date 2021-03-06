from .base import Model
from pathlib import Path
import numpy as np
import joblib

NBR = '(B08 - B11)/(B08 + B11)'
BSI = '((B11 + B04) - (B08 - B02)) / ((B11 + B04) + (B08 - B02))'
NDVI = '(B08 - B04)/(B08 + B04)'

class GeomedianNBR(Model):
    def predict(self, pre, pst):
        pre_nbr = self.eval_expr(NBR, pre)
        pst_nbr = self.eval_expr(NBR, pst)
        return pre_nbr - pst_nbr


class GeomedianBSI(Model):
    def predict(self, pre, pst):
        pre_bsi = self.eval_expr(BSI, pre)
        pst_bsi = self.eval_expr(BSI, pst)
        return pst_bsi - pre_bsi


class GeomedianNDVI(Model):
    def predict(self, pre, pst):
        pre_ndvi = self.eval_expr(NDVI, pre)
        pst_ndvi = self.eval_expr(NDVI, pst)
        return pre_ndvi - pst_ndvi


class GeomedianDiff(Model):
    def predict(self, pre, pst):
        pre_nbr = self.eval_expr(NBR, pre)
        pst_nbr = self.eval_expr(NBR, pst)
        dnbr = pre_nbr - pst_nbr

        pre_bsi = self.eval_expr(BSI, pre)
        pst_bsi = self.eval_expr(BSI, pst)
        dbsi = pst_bsi - pre_bsi

        pre_ndvi = self.eval_expr(NDVI, pre)
        pst_ndvi = self.eval_expr(NDVI, pst)
        dndvi = pst_ndvi - pre_ndvi

        return np.dstack([dnbr, dbsi, dndvi])


class UnsupervisedBurnscarDetect1(Model):
    """
    Unsupervised Burn Scar Detection - Model 1

    This is a simple unsupervised algorithm for detecting burn
    scars in NRT images.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def generate_features(self, pre, pst):
        pre_nbr = self.eval_expr(NBR, pre)
        pst_nbr = self.eval_expr(NBR, pst)

        pre_bsi = self.eval_expr(BSI, pre)
        pst_bsi = self.eval_expr(BSI, pst)

        pre_ndvi = self.eval_expr(NDVI, pre)
        pst_ndvi = self.eval_expr(NDVI, pst)

        return np.dstack([pst_bsi - pre_bsi, pre_ndvi - pst_ndvi, pre_nbr - pst_nbr])

    def predict(self, mask, pre, pst):
        from sklearn.feature_extraction.image import (
            extract_patches_2d,
            reconstruct_from_patches_2d,
        )
        from sklearn import decomposition, cluster
        from skimage import morphology

        X = self.generate_features(pre, pst)

        good = np.isfinite(X).all(axis=2)
        X[~good] = 0

        mu = np.nanmean(X, axis=(0, 1))
        mm = np.nanmedian(X, axis=(0, 1))
        pl, pu = np.nanpercentile(X, (2, 98), axis=(0, 1))
        # TODO parameterise cutoffs

        self.log(f"mean: {mu}")
        self.log(f"median: {mm}")
        self.log(f"percentiles: {(pl, pu)}")

        # TODO: check mu and mm

        for i in range(X.shape[-1]):
            np.clip(X[:, :, i], pl[i], pu[i], out=X[:, :, i])

        # TODO parameterise this
        pX = extract_patches_2d(X, (3, 3))

        oshp = pX.shape

        pX = pX.reshape(pX.shape[0], -1)

        self.log(f"data shape: {pX.shape}")

        # TODO parameterise this
        pca = decomposition.PCA(n_components=5, svd_solver="randomized", whiten=True)
        pX = pca.fit_transform(pX)

        self.log(f"PCA variance: {pca.explained_variance_}")

        self.log(f"reduced shape: {pX.shape}")

        lbls = cluster.KMeans(n_clusters=2).fit_predict(pX)

        y = np.empty(oshp[:3], dtype=np.int8)

        for i in range(lbls.shape[0]):
            y[i, :, :] = lbls[i]

        img = reconstruct_from_patches_2d(y, X.shape[:2])

        self.log(f"class 0: {np.median(X[img == 0], axis=0)}")
        self.log(f"class 1: {np.median(X[img == 1], axis=0)}")

        outclass = np.argmax(
            [
                np.prod(1 + np.median(X[img == 0], axis=0)),
                np.prod(1 + np.median(X[img == 1], axis=0)),
            ]
        )

        self.log(f"outlier class: {outclass}")

        outlier = img == outclass

        mask = morphology.binary_closing(mask, morphology.disk(7))

        outlier = img == outclass
        cloud = mask & outlier

        burnscars = img == outclass
        burnscars[cloud] = 0

        return burnscars


class UnsupervisedBurnscarDetect2(Model):
    """
    Unsupervised Burn Scar Detection - Model 2
    
    This is a simple unsupervised algorithm for detecting burn
    scars in NRT images.
    
    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def generate_features(self, pre, pst):
        pre_nbr = self.eval_expr(NBR, pre)
        pst_nbr = self.eval_expr(NBR, pst)
        dnbr = pre_nbr - pst_nbr

        self.log('DNBR generated')

        pre_bsi = self.eval_expr(BSI, pre)
        pst_bsi = self.eval_expr(BSI, pst)
        dbsi = pst_bsi - pre_bsi

        self.log('DBSI generated')

        pre_ndvi = self.eval_expr(NDVI, pre)
        pst_ndvi = self.eval_expr(NDVI, pst)
        dndvi = pre_ndvi - pst_ndvi

        self.log('DNDVI generated')

        stack = np.dstack([dnbr, dbsi, dndvi, -pst_ndvi])

        self.log('Data stacked')

        return stack

    def predict(self, mask, pre, pst):
        from skimage import feature, draw, morphology
        from sklearn import semi_supervised

        X = self.generate_features(pre, pst)

        self.log(f'Features shape: {X.shape}')

        self.log('Masking data')

        good = np.isfinite(X).all(axis=2)
        X[~good] = 0

        bX = (X[:,:,0]>0)*((1 + X[:, :, 0]) * (1 + X[:, :, 1]) - 1).astype(np.float64)
        #bX = ((1 + X[:, :, 0]) * (1 + X[:, :, 1]) - 1).astype(np.float64)

        bX[mask] = 0

        self.log('Change layer generated')

        # TODO: Parameterise
        with np.errstate(all='ignore'):
            blobs = feature.blob_doh(
                bX, min_sigma=5, max_sigma=30, overlap=0.9, threshold=0.008
            )

        self.log('Blob detection')

        focus = np.zeros_like(bX, dtype=bool)
        for blob in blobs:
            y, x, r = blob
            rr, cc = draw.disk((y, x), r * 5, shape=focus.shape)
            focus[rr, cc] = True

        outliers = np.zeros_like(bX, dtype=np.int8)

        self.log(f"blobs: {blobs.shape[0]}")

        if blobs.shape[0] == 0:
            return outliers[:,:,np.newaxis]

        # TODO: Parameterise radius

        for blob in blobs:
            y, x, r = blob
            rr, cc = draw.disk((y, x), r * 3, shape=outliers.shape)
            outliers[rr, cc] = -1

        for blob in blobs:
            y, x, r = blob
            rr, cc = draw.disk((y, x), r / 2, shape=outliers.shape)
            outliers[rr, cc] = 1

        outliers[mask] = 0

        self.log('Potential outliers masked')

        nunknown = np.count_nonzero(outliers == -1)
        ntotal = np.count_nonzero(focus)

        self.log(f"Spreading: {nunknown} / {ntotal} ({nunknown / ntotal:.4f})")

        y = outliers[focus].reshape((-1,))
        X = X[focus].reshape((-1, X.shape[-1]))

        # TODO: Parametrise
        lblspread = semi_supervised.LabelSpreading(
            kernel="knn", alpha=0.8, max_iter=100, n_neighbors=20, n_jobs=1
        )
        lblspread.fit(X, y)

        self.log(f"Iters: {lblspread.n_iter_}")

        outliers[focus] = lblspread.transduction_
        outliers = outliers.reshape(bX.shape)

        self.log('Done.')

        return outliers[:,:,np.newaxis]


class SupervisedBurnscarDetect1(Model):
    """
    Supervised Burn Scar Detection - Model 1
    
    This is a supervised algorithm for detecting burn
    scars in NRT images.
    
    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = Path(__file__).resolve().parent
        self._model = joblib.load(path / 'burnscar-model-1.pkl')

    def _generate_features(self, pre, pst):
        pre_nbr = self.eval_expr(NBR, pre)
        pst_nbr = self.eval_expr(NBR, pst)

        pre_bsi = self.eval_expr(BSI, pre)
        pst_bsi = self.eval_expr(BSI, pst)

        pre_ndvi = self.eval_expr(NDVI, pre)
        pst_ndvi = self.eval_expr(NDVI, pst)

        return np.dstack([pre_nbr - pst_nbr, pst_bsi - pre_bsi, pst_ndvi - pre_ndvi, pst_ndvi]).reshape((-1, 4))

    def predict(self, mask, pre, pst):
        X = self._generate_features(pre, pst)

        good = np.isfinite(X).all(axis=1)
        X[~good] = 0

        burns = self._model.predict(X).reshape(pre.shape[:2])
        burns[mask] = 0

        from skimage import morphology
        burns = morphology.binary_erosion(burns, morphology.diamond(1))
        #burns = morphology.binary_dilation(burns, morphology.diamond(6))
        #burns = morphology.convex_hull_object(burns)
        #burns = morphology.binary_erosion(burns, morphology.diamond(3))

        return burns[:,:,np.newaxis]

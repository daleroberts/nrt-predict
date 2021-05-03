from .base import Model
import numpy as np


class GeomedianNBR(Model):

    def predict(self, pre, pst):
        # NBR = (B08 - B11)/(B08 + B11)
        pre_nbr = (pre[:, :, 6] - pre[:, :, 8]) / (pre[:, :, 6] + pre[:, :, 8])
        pst_nbr = (pst[:, :, 6] - pst[:, :, 8]) / (pst[:, :, 6] + pst[:, :, 8])

        return pre_nbr - pst_nbr


class GeomedianBSI(Model):

    def predict(self, pre, pst):
        # BSI = ((B11 + B04) - (B08 - B02)) / ((B11 + B04) + (B08 - B02))
        pre_bsi = (pre[:, :, 8] + pre[:, :, 2] - pre[:, :, 6] + pre[:, :, 0]) / (
            pre[:, :, 8] + pre[:, :, 2] + pre[:, :, 6] - pre[:, :, 0]
        )
        pst_bsi = (pst[:, :, 8] + pst[:, :, 2] - pst[:, :, 6] + pst[:, :, 0]) / (
            pst[:, :, 8] + pst[:, :, 2] + pst[:, :, 6] - pst[:, :, 0]
        )

        return pst_bsi - pre_bsi


class GeomedianNDVI(Model):

    def predict(self, pre, pst):
        # NDVI = (B08 - B04)/(B08 + B04)
        pre_ndvi = (pre[:, :, 6] - pre[:, :, 2]) / (pre[:, :, 6] + pre[:, :, 2])
        pst_ndvi = (pst[:, :, 6] - pst[:, :, 2]) / (pst[:, :, 6] + pst[:, :, 2])

        return pre_ndvi - pst_ndvi


class GeomedianDiff(Model):

    def predict(self, pre, pst):
        pre_nbr = (pre[:, :, 6] - pre[:, :, 8]) / (pre[:, :, 6] + pre[:, :, 8])
        pst_nbr = (pst[:, :, 6] - pst[:, :, 8]) / (pst[:, :, 6] + pst[:, :, 8])

        pre_bsi = (pre[:, :, 8] + pre[:, :, 2] - pre[:, :, 6] + pre[:, :, 0]) / (
            pre[:, :, 8] + pre[:, :, 2] + pre[:, :, 6] - pre[:, :, 0]
        )
        pst_bsi = (pst[:, :, 8] + pst[:, :, 2] - pst[:, :, 6] + pst[:, :, 0]) / (
            pst[:, :, 8] + pst[:, :, 2] + pst[:, :, 6] - pst[:, :, 0]
        )

        pre_ndvi = (pre[:, :, 6] - pre[:, :, 2]) / (pre[:, :, 6] + pre[:, :, 2])
        pst_ndvi = (pst[:, :, 6] - pst[:, :, 2]) / (pst[:, :, 6] + pst[:, :, 2])

        diff = np.dstack([pre_nbr - pst_nbr, pst_bsi - pre_bsi, pst_ndvi - pre_ndvi])

        return diff

class UnsupervisedBurnscarDetect1(Model):
    """
    Unsupervised Burn Scar Detection - Model 1
    
    This is a simple unsupervised algorithm for detecting burn
    scars in NRT images.
    
    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """
    
    def log(self, s):
        print(s, file=sys.stderr)
        
    def _generate_features(self, pre, pst):

        pre_bsi = (pre[:, :, 8] + pre[:, :, 2] - pre[:, :, 6] + pre[:, :, 0]) / (pre[:, :, 8] + pre[:, :, 2] + pre[:, :, 6] - pre[:, :, 0])
        pst_bsi = (pst[:, :, 8] + pst[:, :, 2] - pst[:, :, 6] + pst[:, :, 0]) / (pst[:, :, 8] + pst[:, :, 2] + pst[:, :, 6] - pst[:, :, 0])

        pre_ndvi = (pre[:, :, 6] - pre[:, :, 2]) / (pre[:, :, 6] + pre[:, :, 2])
        pst_ndvi = (pst[:, :, 6] - pst[:, :, 2]) / (pst[:, :, 6] + pst[:, :, 2])

        return np.dstack([pst_bsi - pre_bsi, pre_ndvi - pst_ndvi])

    def predict(self, mask, pre, pst):
        from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
        from sklearn import decomposition, cluster
        from skimage import morphology
        
        X = self._generate_features(pre, pst)

        mu = np.nanmean(X, axis=(0,1))
        mm = np.nanmedian(X, axis=(0,1))
        pl, pu = np.nanpercentile(X, (2, 98), axis=(0,1))
        # TODO parameterise cutoffs

        self.log(f"mean: {mu}")
        self.log(f"median: {mm}")    
        self.log(f"percentiles: {(pl, pu)}")
        
        #TODO: check mu and mm

        for i in range(X.shape[-1]):
            np.clip(X[:,:,i], pl[i], pu[i], out=X[:,:,i])

        bad = np.isnan(X)
        X[bad] = 0
        
        # TODO parameterise this
        pX = extract_patches_2d(X, (3, 3))

        oshp = pX.shape
        
        pX = pX.reshape(pX.shape[0], -1)

        self.log(f"data shape: {pX.shape}")
        
        # TODO parameterise this
        pca = decomposition.PCA(n_components=5, svd_solver='randomized', whiten=True)
        pX = pca.fit_transform(pX)

        print(f"PCA variance: {pca.explained_variance_}")

        print(f"reduced shape: {pX.shape}")

        lbls = cluster.KMeans(n_clusters=2).fit_predict(pX)

        y = np.empty(oshp[:3], dtype=np.int8)

        for i in range(lbls.shape[0]):
            y[i,:,:] = lbls[i]

        img = reconstruct_from_patches_2d(y, X.shape[:2])

        outclass = np.argmax([np.prod(1+np.mean(X[img==0], axis=0)), np.prod(1+np.mean(X[img==1], axis=0))])
        
        self.log(f"outlier class: {outclass}")

        outlier = img == outclass

        mask = morphology.binary_closing(mask, morphology.disk(7))

        outlier = img == outclass
        cloud = mask & outlier

        burnscars = img == outclass
        burnscars[cloud] = 0
        
        return burnscars

from .base import Model


class GeomedianNBR(Model):

    def predict(self, datas):
        print("predicting")
        pre = datas[0]
        pst = datas[1]

        # NBR = (B08 - B11)/(B08 + B11)
        pre_nbr = (pre[:, :, 6] - pre[:, :, 8]) / (pre[:, :, 6] + pre[:, :, 8])
        pst_nbr = (pst[:, :, 6] - pst[:, :, 8]) / (pst[:, :, 6] + pst[:, :, 8])

        return pre_nbr - pst_nbr


class GeomedianBSI(Model):

    def predict(self, datas):
        pre = datas[0]
        pst = datas[1]

        # BSI = ((B11 + B04) - (B08 - B02)) / ((B11 + B04) + (B08 - B02))
        pre_bsi = (pre[:, :, 8] + pre[:, :, 2] - pre[:, :, 6] + pre[:, :, 0]) / (
            pre[:, :, 8] + pre[:, :, 2] + pre[:, :, 6] - pre[:, :, 0]
        )
        pst_bsi = (pst[:, :, 8] + pst[:, :, 2] - pst[:, :, 6] + pst[:, :, 0]) / (
            pst[:, :, 8] + pst[:, :, 2] + pst[:, :, 6] - pst[:, :, 0]
        )

        return pst_bsi - pre_bsi


class GeomedianNDVI(Model):

    def predict(self, datas):
        pre = datas[0]
        pst = datas[1]

        # NDVI = (B08 - B04)/(B08 + B04)
        pre_ndvi = (pre[:, :, 6] - pre[:, :, 2]) / (pre[:, :, 6] + pre[:, :, 2])
        pst_ndvi = (pst[:, :, 6] - pst[:, :, 2]) / (pst[:, :, 6] + pst[:, :, 2])

        return pre_ndvi - pst_ndvi


class GeomedianDiff(Model):

    def predict(self, datas):
        pre = datas[0]
        pst = datas[1]

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


class GeomedianDiffSingle(Model):

    def predict(self, datas):
        pre = datas[0]
        pst = datas[1]

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

        #np.clip(diff, -1, 1, out=diff)

        img = np.prod(diff, axis=2)

        from skimage.feature import hessian_matrix_det

        img = hessian_matrix_det(img.astype(np.float64), sigma=0.5, approximate=True)

        return img.astype(np.float32)


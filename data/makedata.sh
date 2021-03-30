#!/usr/bin/env zsh

# Make some *small* test data

PKG='/2021-02-05/S2A_OPER_MSI_ARD_TL_VGS1_20210205T055002_A029372_T50HMK_N02.09/'

mkdir -p test

aws --no-sign-request s3 sync s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD$PKG ./test/${PKG:t}

for f in ./test/${PKG:t}/LAMBERTIAN/*; do
    echo ${f}
    rm ${f}
    touch ${f}
done

for f in ./test/${PKG:t}/SUPPLEMENTARY/*; do
    echo ${f}
    rm ${f}
    touch ${f}
done

for f in ./test/${PKG:t}/NBART/*; do
    echo ${f}
    rm ${f}
    ff=${f:t}
    ln -s ../NBAR/${ff:gs/NBART/NBAR/} $f
done

for f in ./test/${PKG:t}/QA/*CONTIGUITY*; do
    rm ${f}
    touch ${f}
done

for f in ./test/${PKG:t}/QA/*SHADOW*; do
    rm ${f}
    touch ${f}
done

for f in ./test/${PKG:t}/NBAR/*.TIF; do
    echo ${f}
    gdal_translate -outsize 1% 1% $f ${f:h}/_${f:t}
    mv ${f:h}/_${f:t} ${f:h}/${f:t}
done

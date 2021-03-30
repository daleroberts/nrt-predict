Analysis Ready Data (ARD) Package Contents Outline
==================================================

ARD packages are structured as follows:

<Granule ID>/ # Granule ID is akin to Scene ID for Landsat and Tile ID for Sentinel-2
    
    - README.md # The contents of this file
    - ARD-METADATA.yaml
    - ARD-METADATA.xml # An ISO-19115 xml schema of ARD-METADATA.yaml (To be produced in a BETA release.)
    - <Granule ID>_FMASK.TIF # The output of the Fmask algorithm
    - map.html # A web browser interactive map representing the valid data extents of the Granule
    - bounds.geojson # A GeoJSON file containing the vertices of the valid data extents of the Granle
    - CHECKSUM.sha1 # Contains the SHA1 checksum for each of the files contained in this ARD package

    QA/ # Datasets related to quality assurance and/or categorisation

        - <Granule ID>_FMASK.TIF # The output of the Fmask algorithm
        - <Granule ID>_COMBINED_TERRAIN_SHADOW.TIF # Self, Cast (direction of sun & satellite) shadow
        - <Granule ID>_LAMBERTIAN_CONTIGUITY.TIF # Contiguity determined from lambertian products (if available)
        - <Granule ID>_NBAR_CONTIGUITY.TIF # Contiguity determined from nbar products
        - <Granule ID>_NBART_CONTIGUITY.TIF # Contiguity determined from nbart products
        - <Granule ID>_SBT_CONTIGUITY.TIF # Contiguity determined from sbt products (if available)

    SUPPLEMENTARY/ # Supplementary datasets used in the creation of surface reflectance and surface brightness temperature
        - <Granule ID>_SATELLITE_VIEW.TIF
        - <Granule ID>_SATELLITE_AZIMUTH.TIF
        - <Granule ID>_SOLAR_ZENITH.TIF
        - <Granule ID>_SOLAR_AZIMUTH.TIF
        - <Granule ID>_RELATIVE_AZIMUTH.TIF
        - <Granule ID>_TIMEDELTA.TIF # Time from apogee in seconds
        - <Granule ID>_INCIDENT.TIF
        - <Granule ID>_AZIMUTHAL_INCIDENT.TIF
        - <Granule ID>_EXITING.TIF
        - <Granule ID>_AZIMUTHAL_EXITING.TIF
        - <Granule ID>_RELATIVE_SLOPE.TIF

    NBAR/ # BRDF corrected surface reflectance

        - <Image ID>_NBAR_<Band ID>.TIF
        - <Image ID>_NBAR_QUICKLOOK.TIF # Colour enhanced true colour composite
        - <Image ID>_NBAR_THUMBNAIL.JPG # JPEG version of the QUICKLOOK

    NBART/ # BRDF & Terrain corrected surface reflectance

        - <Image ID>_NBART_<Band ID>.TIF
        - <Image ID>_NBART_QUICKLOOK.TIF # Colour enhanced true colour composite
        - <Image ID>_NBART_THUMBNAIL.JPG # JPEG version of the QUICKLOOK

    LAMBERTIAN/ # Surface reflectance (only supplied if required)

        - <Image ID>_LAMBERTIAN_<Band ID>.TIF
        - <Image ID>_LAMBERTIAN_QUICKLOOK.TIF # Colour enhanced true colour composite
        - <Image ID>_LAMBERTIAN_THUMBNAIL.JPG # JPEG version of the QUICKLOOK

    SBT/ # Surface brightness temperature (only supplied if available and required)

        - <Image ID>_SBT_<Band ID>.TIF


Band Alias
----------

Alias names for a given <Band ID> are determined via:
[wagl](https://github.com/GeoscienceAustralia/wagl/blob/develop/wagl/acquisition/sensors.json)


Fmask Classification Lookup
---------------------------

* 0 -> Null/Fill Value
* 1 -> Valid
* 2 -> Cloud
* 3 -> Cloud Shadow
* 4 -> Snow
* 5 -> Water


ADDITIONAL NOTES
----------------

* File naming conventions are inherited from the parent L1 dataset.
* TIF files are generated using a cloud optimised GeoTIFF creation method
* The FMASK image was generated using the pythonfmask module available at http://pythonfmask.org
* The overviews are built with Nearest Neighbour resampling, where possible. The overviews for the FMASK file are built via pythonfmask
* The number of bands available in a given product i.e. *NBAR*, depend on the supported attribute given in:
[wagl](https://github.com/GeoscienceAustralia/wagl/blob/develop/wagl/acquisition/sensors.json)
* If SBT is availble quicklooks and thumbnails are currently not generated as the bulk of the products will be single band anyway.
* Datasets containing angles such as SATELLITE_VIEW, will be expressed in degress, not radians.

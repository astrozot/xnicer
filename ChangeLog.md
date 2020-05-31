# Change Log

## Version 0.1

*Released on 13 May 2019*

Initial release.

## Version 0.2

*Partial on 31 May 2020*

- Computation of the intrinsic colors and associated covariance 
  matrix.
- Addition of attributes to use different classes of objects in
  photometric and color catalogues.
- A PhotometricCatalogue can now be built automatically from a VOTable
- Use of Cython code to perform the XD and the prediction
- Change of the XDCV class name into XD_Mixture
- Several tests added in the file test_em_step
- Better file organization
- Moving the reddening_law vector to the PhotometricCatalogue class
- Using warnings for warnings :-)
- Classes of different objects can now be used to improve the XD
- Using children of Table for all catalogs
- Implemented fuzzy class in xdeconv and xnicer
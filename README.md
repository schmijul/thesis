# student thesis : estimation of high resolution radio maps

 This repo features a few scripts for spatial interpoaltion and deep learning methods for estimating radio maps


 ### To get started :
 
 Since a few different libraries are used, you need to install them first.

 run:
    ```
    pip install -r requirements.txt
    ```
### Pipeline

The pipeline is divided into two parts.

#### Code Quality Check

This part checks the quality of the code using pylint.

The Pipeline will only accept pull request with a 9/10 or higher code quality rating.

#### Unit Tests

This part will check if the main data processing Functions still work

### Interpolation

To use basic interpolation techniques on griddata and kriging we use the functions stored in *interpolation_uitls.py*

The script interpolation uses said functions to estimates either specific points or the entire map based on a few known points.

##### Linear Interpolation

Interpolation on griddata is handled by scipys *griddata* function.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

Here it is mostly used to handle linear interpolation, but cubic or higher interpolyation polynoms are also possible.

##### Kriging

To handle kriging we use the scikit-geostats package.

https://scikit-gstat.readthedocs.io/en/latest/

This library is supports the creation of variograms, regressionvariograms and execute ordinary kriging with a given variogram.



#### Machine Learning

All machine learning/ deep learning tasks are handled by tensorflow & keras.

There are two scripts for this one named *basemodel.py*.
This one will create a model name basemdoel which uses x and y coordinates as an input and is trained to predict one numerical value as an output.

The other one is *deepkriging.py* which will apply the wendland kernel onto a given dataset, which will create more than just two input dimensions.
note:
This can cause hughe amount of trainable parameters and memory usage.
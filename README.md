# Content-Based Image Retrievl with custom k-nearet Neighbors
This project demonstrate a simple content-based image retrievel(CBIR) system using a k-nearest neighbors algorithm. The Algorithm retrieves similar images froma dataset based on the color histograms f the query image and the dataset images. The custom KNN algorithm uses chi-square distance as a metyric to find the similar images.

###Dependencies
-Python 3.7+
-opencv
-Numpy

You can install the requirted libraries with following command:
pip install opencv-python numpy

###How to use
1. Clone the repo
2. Place your dataset of images in the folder name 'Object_data' in the project.
3. PLace the query image in the folder named 'query'
4. Run the script, 'cbir_knn.py'

python cbir_knn.py

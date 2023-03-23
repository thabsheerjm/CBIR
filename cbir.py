import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors


def dataset(dataset_path):
    images = []
    labels = []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path,folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            if file_name.endswith('.jpg'):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    labels.append(folder_name)

    return images, labels 



def extract_color_histogram(img, bins =(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(histogram,histogram)
    
    return histogram.flatten()

def chi_squared_distance(hist_A, hist_B, eps =1e-10):
    return (1/2)*np.sum([((a-b)**2)/(a+b+eps) for a,b in zip(hist_A,hist_B)]) 


imgs, labls = dataset('./256_ObjectCategories/')
feature_vectors = [extract_color_histogram(img) for img in imgs]

k =10
knn = NearestNeighbors(n_neighbors=k, metric = chi_squared_distance)
knn.fit(feature_vectors)

def retrieve_similar_images(query_image,images, knn_model, k=10):
    query_feature_vector = extract_color_histogram(query_image)
    indices = knn_model.kneighbors([query_feature_vector],n_neighbors=k, return_distance= False)
    return [images[idx] for idx in indices[0]]

query_image = cv2.imread('./query/image.jpg')
retrieved_images = retrieve_similar_images(query_image,imgs,knn)


for i, img in enumerate(retrieved_images):
    cv2.imshow(f"Retrieved image matches {i+1}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
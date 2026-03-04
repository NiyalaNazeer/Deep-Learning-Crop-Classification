 # Deep Learning-Based Crop Classification Using Hyperspectral Images

This repository presents a deep learning-based approach for crop classification using hyperspectral and multi-temporal satellite imagery. Accurate crop classification plays a critical role in precision agriculture, enabling better yield estimation, crop monitoring, irrigation planning, and agricultural decision-making.

Hyperspectral imagery captures reflectance information across numerous narrow spectral bands, allowing detailed analysis of vegetation characteristics such as chlorophyll content, moisture levels, and plant health. Unlike traditional RGB imagery, hyperspectral data provides high-dimensional spectral signatures that enable fine-grained discrimination between crop types.

With the availability of high-resolution and high-frequency satellite observations, multi-temporal data allows modeling crop growth patterns across time. This temporal information improves classification robustness and helps distinguish crops with similar spectral responses at a single time point.

Traditional crop classification approaches rely on mono-temporal imagery and handcrafted spectral or texture-based features. While computationally efficient, these methods often struggle with high-dimensional nonlinear patterns present in hyperspectral data. 

In this project, we implement and compare classical machine learning models with deep learning-based models to evaluate their effectiveness in hyperspectral crop classification. The Convolutional Neural Network (CNN) model automatically learns hierarchical feature representations directly from spectral inputs, reducing the need for manual feature engineering and improving classification performance.

This project demonstrates the application of Artificial Intelligence and Deep Learning in remote sensing and agricultural analytics.

---

## Dataset

The dataset used in this project was obtained from Mendeley Data:

https://data.mendeley.com/datasets/3j5w87djyh/1

The dataset consists of hyperspectral and multi-band satellite imagery along with labeled crop classes for supervised learning. It provides high-dimensional spectral information that enables detailed crop discrimination based on reflectance characteristics.
 
## Steps

1. Prepare and preprocess the dataset  
2. Extract spectral indices and temporal features  
3. Train classical machine learning models  
4. Train the CNN-based deep learning model  
5. Evaluate model performance using accuracy and confusion matrix  
6. Compare results across models  

---

## Results

- The Deep Learning model achieved higher classification accuracy compared to classical machine learning approaches.  
 

## License

This project is licensed under the MIT License.




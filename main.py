import streamlit as st

# Title and Team Information
st.title('CS7641: Project Proposal')
st.subheader('(Group 42)')
st.header('Tumor Detection and Classification')
st.markdown('**Team Members:** Shital Salke, Jenny Lin, Koushika Kesavan, Hima Varshini Parasa')
st.markdown('**Institution:** Georgia Institute of Technology')

# Introduction and Background Section
st.header('1. Introduction and Background')
st.write("""
Brain tumors are abnormal cell growths in or around the brain, classified as benign (non-cancerous) or malignant (cancerous). In adults, the primary types include Gliomas (usually malignant), Meningiomas (often benign), and Pituitary tumors (generally benign). Early diagnosis is crucial for improving outcomes and treatment options. This project aims to develop a machine learning model for detecting and classifying brain tumors using MRI images.

Several studies highlight the effectiveness of modern techniques. A YOLOv7-based model with attention mechanisms achieved 99.5% accuracy [1]. Transfer learning models like VGG16 reached 98% accuracy, outperforming traditional CNNs [2]. A CNN optimized with the Grey Wolf Optimization algorithm achieved 99.98% accuracy [3], while MobileNetv3 reached 99.75%, surpassing ResNet and DenseNet [4]. Additionally, one study compared CNNs (96.47%) with Random Forest (86%) for tumor classification [5].

The dataset for this project, “Brain Tumor (MRI scans),” sourced from Kaggle, contains 3,264 MRI images across three tumor types: Gliomas, Meningiomas, and Pituitary tumors, with a balanced distribution of images in various orientations.
""")
st.markdown('Link to the Dataset: [https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data)')

# Problem Definition Section
st.header('2. Problem Definition')
st.write("""
Manual analysis of MRI scans by radiologists is labor-intensive and error-prone. This project aims to automate the detection and classification of brain tumors (Gliomas, Meningiomas, and Pituitary tumors) to improve accuracy and efficiency compared to existing models.

The solution involves developing a deep learning model using a custom CNN with multiple convolutional, pooling, and fully connected layers. Optimization will utilize Transfer Learning with models like VGG16, ResNet50, and EfficientNet. Performance will be assessed using metrics such as Precision, Recall, Accuracy, F1 Score, and ROC-AUC.

**Difference From Prior Literature**: Our approach enhances existing methods by utilizing EfficientNetB2 for effective feature extraction and incorporating unsupervised techniques like DBSCAN and GMM to address noise and irregular tumor shapes. This combination leads to improved model performance and faster, more accurate tumor classification.
""")

# Methods Section
st.header('3. Methods')
st.subheader('Data Preprocessing Methods')
st.write("""
To prepare MRI images for effective model training, preprocessing is crucial to ensure clean, relevant, and balanced data. The following techniques will be used:
- **Data Cleaning**: Filters will remove noise and artifacts from MRI scans, allowing the model to focus on relevant features and improving accuracy.
- **Dimensionality Reduction**: Techniques like PCA will reduce image size without losing key information, facilitating faster training and minimizing overfitting.
- **Data Augmentation**: Techniques such as rotations, flips, and zooms will enhance the dataset, addressing class imbalance and increasing data diversity for better model generalization.
""")

st.subheader('Machine Learning Algorithms')
st.write("""
- **Unsupervised Learning**:
  - **KMeans**: Clusters similar data points to reveal patterns.
  - **DBSCAN**: Identifies irregularly shaped tumors while managing noise.
  - **GMM**: Models tumor regions as Gaussian distributions for segmentation.

- **Supervised Learning**:
  - **EfficientNetB2**: Pre-trained model excelling in feature extraction, balancing accuracy and efficiency.
  - **ResNet50**: Deep network that addresses the vanishing gradient problem for improved accuracy.
  - **VGG16**: Effective deep network for high accuracy in image classification.
  - **CNN**: Learns tumor features from images for optimal classification [6].
  - **SVM**: Effectively separates tumor classes in small datasets.
  - **Random Forest**: Combines decision trees for robust classifications.
""")

# Potential Results and Discussion Section
st.header('4. (Potential) Results and Discussion')
st.subheader('ML Metrics')
st.write("""
We will evaluate our results using the following metrics [7]:
- **F1 Score**: A balanced measure of precision and recall, crucial in clinical settings where false positives and negatives have serious implications.
- **AUC-ROC**: Represents the model's discriminative power across all classification thresholds.
- **Confusion Matrix**: Provides detailed performance insights for each tumor type.
- **Cross-Validation**: K-fold cross-validation ensures model consistency and generalizability across diverse patient data.
""")
st.subheader('Project Goals')
st.write("""
- Detect and classify brain tumors through MRI scans.
- Reduce false positives and detect tumors early.
- Generate consistent accuracies for different types of brain tumors.
- Generalize well to unseen MRI scans.
""")

st.subheader('Expected Results')
st.write("""
- **F1 Score**: 97-99%
- **AUC-ROC**: Above 0.95
- **Confusion Matrix**: High positive rates for various tumor types
- **Cross-validation**: Less than 1-2% standard deviation in accuracy across folds
""")

# Gantt Chart Section
st.header('5. Gantt Chart')
st.image('./gantt.png')

# Contribution Table Section
st.header('6. Contribution Table')
st.write("""
| **Team Member** | **Contributions** |
|-----------------|----------------------------|
| **Koushika**    | Introduction and Background, Problem Definition, References, Brainstorming project proposal ideas, Recording the video presentation |
| **Jenny**       | Methods, IEEE Citations, Brainstorming project proposal ideas, Putting together Google slides for video presentation |
| **Hima**        | (Potential) Results + Discussion, References, Brainstorming project proposal ideas, Putting together Google slides for video presentation |
| **Shital**      | Gantt Chart, References, Brainstorming project proposal ideas, Setting up Streamlit share for hosting the project proposal page, Github project setup and access |
""")

# References Section
st.header('7. References')
st.write("""
[1] A. B. Abdusalomov, M. Mukhiddinov, and T. K. Whangbo, “Brain tumor detection based on deep learning approaches and Magnetic Resonance Imaging,” *Cancers*, [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10453020/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10453020/)(accessed Oct. 4, 2024).\n
[2] M. Z. Khaliki and M. S. Başarslan, “Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN,” *Nature News*, [https://www.nature.com/articles/s41598-024-52823-9](https://www.nature.com/articles/s41598-024-52823-9)  (accessed Oct. 4, 2024).\n
[3] H. ZainEldin et al., “Brain tumor detection and classification using deep learning and sine-cosine fitness grey wolf optimization,” *Bioengineering (Basel, Switzerland)*, [https://pubmed.ncbi.nlm.nih.gov/36671591/](https://pubmed.ncbi.nlm.nih.gov/36671591/) (accessed Oct. 4, 2024).\n
[4] S. K. Mathivanan et al., “Employing deep learning and transfer learning for accurate brain tumor detection,” *Nature News*, [https://www.nature.com/articles/s41598-024-57970-7](https://www.nature.com/articles/s41598-024-57970-7). (accessed Oct. 4, 2024)\n
[5] S. Saeedi, S. Rezayi, H. Keshavarz, and S. R. N. Kalhori, “MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques - BMC Medical Informatics and decision making,” *BioMed Central*, [https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6) (accessed Oct. 4, 2024). \n
[6] B. Babu Vimala et al., “Detection and classification of brain tumor using hybrid deep learning models,” *Scientific Reports*, [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10754828/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10754828/) (accessed Oct. 4, 2024).\n
[7] J. Amin, M. Sharif, A. Haldorai, M. Yasmin, and R. S. Nayak, “Brain tumor detection and classification using Machine Learning: A comprehensive survey - complex & intelligent systems,” *SpringerLink*, [https://link.springer.com/article/10.1007/s40747-021-00563-y](https://link.springer.com/article/10.1007/s40747-021-00563-y) (accessed Oct. 4, 2024).\n
""")

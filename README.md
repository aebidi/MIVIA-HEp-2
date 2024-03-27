
# Classification of HEp-2 staining patterns

The task is to develop an image processing /machine learning algorithm to detect cell images in a 
whole slide of HEp2 Image. The actual Labels marked by a specialist are given in the dataset. 
The results of your algorithm shall be evaluated via standard performance metrics discussed 
below. This involves matching your resultant bounding boxes around cells with labeled masks 
via the formulas.

Referring to the problem statement above, we have first applied the preprocessing techniques on 
the dataset and then developed a machine learning model to incorporate the problem hence 
have evaluated the performance matrix including sensitivity, specificity, and overall accuracy of 
the propped model for the problem statement.

![labels](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/260a88cb-fadd-463b-949b-dfdaeac0e4d6)



## Algorithmic Pipeline

The algorithmic pipeline followed is:

•	First import the related libraries for the preprocessing techniques for the problem such as NumPy, OpenCV etc.

•	Load images from the directory containing the dataset with .bmp extension.

•	Apply the image pre-processing techniques to the dataset (segmentation)

    a)	First, we will convert colored images to grayscale.

    b)	Apply Binary thresholding plus OTSU thresholding to obtain a binary image.

    c)	Use CV2.morphologyx to fill in holes in the image.

    d)	Apply distance transform to achieve the best result by defining more details.

    e)	Then again use thresholding to convert the grayscale image produced after morphologyx operation into a binary image.

•	Make 2 paths (Directories) one for the preprocessed images (masks created by us) as derived from the above step and the other for the masks that were given in the project dataset.

•	Calculate the mean square error between the preprocessed images and the masks.

•	Prepared CSV file containing the labels of the given dataset along with the preprocessed images.

•	Now for the machine learning part make a data frame containing the images and their labels.

•	Split train and test for the proposed CNN model.

•	Now fit the model onto the dataset. 

•	Make the confusion matrix and calculate the performance matrix for the given problem using the formula given (precision, recall, F1-score, sensitivity, specificity, and accuracy).

## Performance Evaluation

![metrics](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/e1d23dd1-9b2b-423e-931a-3d90252542ea)

The following code below shows all the performance metrics with 89% accuracy which is considered quite acceptable. The Recall = 1 also is a good indication. 0.9438 is a good F1-score means that you have low false positives and low false negatives.
## Implementation

1.	Loading Data:

•	The code first loads two types of data: full images with their corresponding full masks and individual cell images with their corresponding cell masks.
•	It iterates over each folder in the 'Images' directory and reads each full image and its corresponding full mask.
•	Similarly, it iterates over each folder in the 'Cells/Cell_Images' directory and reads each individual cell image along with its corresponding cell mask.



2.	Drawing Boundaries and Calculating Metrics:

•	The draw_boundaries function takes the loaded full images, full masks, individual cell images, and individual cell masks as input.
•	For each image, it draws boundaries around cells using contours derived from the individual cell masks.
•	It then compares the individual cell masks with the full mask to calculate performance metrics such as true positives, false positives, false negatives, and true negatives.

3.	Performance Metrics Calculation:

After drawing boundaries and calculating metrics for all images, the code calculates the following performance metrics:

•	Accuracy: The proportion of correctly classified cells among all cells.
•	Precision: The proportion of correctly classified positive cells among all cells classified as positive.
•	Recall: The proportion of correctly classified positive cells among all actual positive cells.
•	F1-score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

4.	Displaying Images and Results:

•	The code displays images with cell boundaries using the OpenCV cv2.imshow function.
•	After processing all images, it prints the accuracy, precision, recall, and F1-score to the console.

5.	Result of Performance Metrics

•	Precision ≈ 0.89
•	Recall = 1.0
•	F1-score ≈ 0.9438
•	Accuracy = 89%
By comparing the detected boundaries with manually annotated ground truth boundaries, these metrics can quantify the algorithm's performance in terms of accuracy and reliability.

## Visual Representation of the Data

After segmentation, the following type of dataset is obtained from the 28 images. Here, only a few of those images are shown:


![data1](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/499c4a26-bab3-47f0-b407-37034025eba1)

![data2](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/be2dd345-ee8d-4858-a750-640f9e2d3149)

![data3](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/02fea111-953c-4530-80db-b739addcadad)

![data4](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/01bdc984-21c2-4ece-bf8c-a44e8022a712)

![data5](https://github.com/aebidi/MIVIA-HEp-2/assets/89690384/0c0c109a-18f2-4618-9b94-9ec4f87ff76a)



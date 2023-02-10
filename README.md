# AWS-Tutorials
Collection of tutorials involving use of ML models on AWS platform

1. CD Enrollment Prediction (Build, Train, and Deploy a Machine Learning Model with Amazon SageMaker)

In this tutorial I have used Amazon SageMaker to build, train, and deploy a machine learning (ML) model using the XGBoost ML algorithm

In this tutorial, I develop a machine learning model to predict whether a customer will enroll for a certificate of deposit (CD).

Below are the steps involved in this tutorial

- Create a SageMaker notebook instance
- Prepare the data
- Train the model to learn from the data
- Deploy the model
- Evaluate your ML model's performance

Dataset: Bank Marketing Data Set (contains information on customer demographics, responses to marketing events, and external factors. The data has been labeled for your convenience, and a column in the dataset identifies whether the customer is enrolled for a product offered by the bank. A version of this dataset is publicly available from the Machine Learning Repository curated by the University of California, Irvine.)

## Steps

0. Signup for an AWS account or Signin to AWS account if you already have one
1. Create an Amazon SageMaker notebook instance
  - Sign in to the Amazon SageMaker console, and in the top right corner, select your preferred AWS Region. This tutorial uses the US West (Oregon) Region.
  - In the left navigation pane, choose Notebook instances, then choose Create notebook instance.
  - On the Create notebook instance page, in the Notebook instance setting box, fill the following fields:
    * For Notebook instance name, type SageMaker-Tutorial.
    * For Notebook instance type, choose ml.t2.medium.
    * For Elastic inference, keep the default selection of none.
    * For Platform identifier, keep the default selection.
  - In the Permissions and encryption section, for IAM role, choose Create a new role, and in the Create an IAM role dialog box, select Any S3 bucket and choose
    * Create role.
    (Note: If you already have a bucket that you’d like to use instead, choose Specific S3 buckets and specify the bucket name.)
  - Keep the default settings for the remaining options and choose Create notebook instance.
    (In the Notebook instances section, the new SageMaker-Tutorial notebook instance is displayed with a Status of Pending. The notebook is ready when the Status changes to InService.)
2. Prepare the data
  - After your SageMaker-Tutorial notebook instance status changes to InService, choose Open Jupyter.
  - In Jupyter, choose New and then choose conda_python3.
  - In a new code cell on your Jupyter notebook, copy and paste the following code and choose Run.
  - Create the S3 bucket to store your data. Copy and paste the following code into the next code cell and choose Run.
    (Note: Make sure to replace the bucket_name your-s3-bucket-name with a unique S3 bucket name. If you don't receive a success message after running the code           change the bucket name and try again.)
  - Download the data to your SageMaker instance and load the data into a dataframe. Copy and paste the following code into the next code cell and choose Run.
  - Shuffle and split the data into training data and test data. Copy and paste the following code into the next code cell and choose Run.
    The training data (70% of customers) is used during the model training loop. You use gradient-based optimization to iteratively refine the model parameters. 
     Gradient-based optimization is a way to find model parameter values that minimize the model error, using the gradient of the model loss function. The test data       (remaining 30% of customers) is used to evaluate the performance of the model and measure how well the trained model generalizes to unseen data.


3. Train the ML model
  - Run the code block in model.py
  - Set up the Amazon SageMaker session, create an instance of the XGBoost model (an estimator), and define the model’s hyperparameters. Copy and paste the               following code into the next code cell and choose Run.
  - Start the training job. Copy and paste the following code into the next code cell and choose Run.
      This code trains the model using gradient optimization on a ml.m4.xlarge instance. After a few minutes, you should see the training logs being generated in           your Jupyter notebook.
 4. Deploy the model
    In this step, you deploy the trained model to an endpoint, reformat and load the CSV data, then run the model to create predictions.
  
  - In a new code cell on your Jupyter notebook, copy and paste the following code and choose Run.
  - To predict whether customers in the test data enrolled for the bank product or not, go next code cell and choose Run. 
 
 5. Evaluate model performance
    In this step, you evaluate the performance and accuracy of the machine learning model.
 
 6. Clean up
    In this step, you terminate the resources you used in this lab.
    Important: Terminating resources that are not actively being used reduces costs and is a best practice. Not terminating your resources will result in charges to       your account.
   - Delete your endpoint
   - Delete your training artifacts and S3 bucket
   - Delete your SageMaker Notebook 


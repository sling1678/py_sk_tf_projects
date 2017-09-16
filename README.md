# py_sk_tf_projects
Here I am trying to write software for ML projects in a more general way. Every project appears to have same type of steps.
(1) Data acquiring - usually fetch data from some website, extracting, converting to some DataFrame for easy manipulation
(2) Create a Data cleaning pipeline 
  - filling the NaNs 
  - plotting features against the labels
  - discover important features
      - try PCA
      - try Decision Tree
  - discover any non-linear combination
  - building a pipeline
(3) Experimenting with three or four ML algorithms on a smaller subset of data - I will start with sklearn and then move to TensorFlow
  - A linear classifier/regressor
  - Definitely a Tree - maybe boosted random forest algorithm
  - SVM with rbf kernel trick
  - Deep ANN
(4) Aplying the full pipeline of leading ML algorithms to the full data separated into training, vaidation and testing subsets
(5) Choose the best performer - be sure why
(6) Write a report

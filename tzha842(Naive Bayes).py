import numpy as np
class NaiveBayes:
    def __init__(self):
        self.X_train, self.y_train = [], []
        self.classes = np.array(['A', 'B', 'E', 'V'])
        
        self.unique_words = {}  # key: unique_word in the entire file. value: frequency of that word
        self.frequent_words = []
        
        self.priors = None
        self.likelihoods = None
        
        
    def read(self, training_file, testing_file):
        """
        Read all the data from 3 files. Process the file into datasets and set the corresponding attribute.
        """
        with open(training_file, 'r') as train, open(testing_file, 'r') as test:   
            # Skip first line
            train_lines = train.readlines()[1:]  
            test_lines = test.readlines()[1:]
        
        """###########################  Training Data  ###########################"""
        n_samples = len(train_lines)
        n_words = 1000  # how many feature the modle will use
        
        y_train = []
        X_train = []
        
        self.vocabulary = [] # Vocabulary of every abstract from the file. With each element being a dict of words from each abstract

        for line in train_lines:
            segment = line.rstrip().split(',')
            y = segment[1]
            words = segment[2].replace('\"', '').split(' ') # Remove the quotation mark "
            
            # Remove any unnecessary word
            new_words = []
            vocabulary = {}  # Unique word in this abstract with freq
            for word in words:
                """
                Preprocess the data to remove unnecessary words.
                """ 
                if len(word) <= 4:
                    continue
                if len(word) >= 9:
                    continue
                if word.isdigit() or any(char.isdigit() for char in word):
                    continue
                if any(word in existing_word for existing_word in new_words):  # Check if word is substring of any existing word
                    continue
                new_words.append(word)
                vocabulary[word] = vocabulary.get(word, 0) + 1  # Add one count to current vocabulary
                self.unique_words[word] = self.unique_words.get(word, 0) + 1  # Unique word in the entire file

            X = np.array(new_words)
            self.vocabulary.append(vocabulary)
            
            y_train.append(y)
            X_train.append(X)
        
        self.y_train = np.array(y_train)
        # Identify the 1000 most frequently occurring words after preprocessing and generate 0-1  
        # attributes stating whether or not the word occurs in the corresponding abstract.
        self.frequent_words = [x[0] for x in sorted(self.unique_words.items(), key=lambda x: x[1], reverse=True)][:n_words]
        # Init to be a n_samples by n_words zero matrix
        self.X_train = np.zeros((n_samples, n_words))
        # Check whether a frequent word appear in the corresponding 
        # abstract, set the cooresponding attribute to its frequency
        for i in range(n_samples):
            for j in range(n_words):
                if self.frequent_words[j] in X_train[i]:
                    self.X_train[i][j] = self.vocabulary[i][self.frequent_words[j]]
        
        """###########################  Testing Data  ###########################"""

        X_test = []
        
        # Process the lines from the input testing file
        for line in test_lines:
            words = line.rstrip().split(',')[1].replace('\"', '').split(' ')
            # Init X to be a zero vector of n_words attributes
            X = np.zeros(n_words)
            
            for word in words:    
                if len(word) <= 4:
                    continue
                if len(word) >= 9:
                    continue
                if word.isdigit() or any(char.isdigit() for char in word):
                    continue
                # if a word is in the frequent word list, increment its attribute value
                for j in range(n_words):
                    if word == self.frequent_words[j]:
                        X[j] += 1
            # Done processing one line
            X_test.append(X)
        
        # Entire testing data
        self.X_test = np.array(X_test)
        

    def train(self, X=None, y=None):
        """Arguments:
            X: X_training type: 2d np.array
            y: y_training type: 1d np.array
           Generate priors and likelihoods from the training examples.
        """
        # Set default values
        if X is None: X = self.X_train
        if y is None: y = self.y_train
        
        n_samples, n_words = X.shape
        n_classes = 4
        
        # Init priors and likelihoods
        self.priors = np.zeros(n_classes, dtype=np.float64)
        self.likelihoods = np.zeros((n_classes, n_words), dtype=np.float64)
        
        for i, c in enumerate(self.classes):
            X_c = X[c==y]  # Select all instances X with class being c
            self.priors[i] = X_c.shape[0] / float(n_samples)
            self.likelihoods[i] = (X_c.sum(axis=0) + 1) / (float(X_c.sum() + 1000))  # 1000 being total number of attributes   

     
    def _predict(self, x):
        """
        Run the classifer on one single instance x.
        Return the prediction class.
        """
        posteriors = np.zeros(4)  # Empty posterior probability for four differen classes
        
        for i, c in enumerate(self.classes):
            # Doing calculation in log space
            prior = np.log(self.priors[i])
            likelihood = np.dot(x, np.log(self.likelihoods[i]))
            # likelihood = np.sum(np.log(self.likelihoods[i]))
            posteriors[i] = prior + likelihood
        return self.classes[np.argmax(posteriors)]
    
    
    def test(self, filename_to_write):
        """
        A wrapper function to run the method _predict() on every single 
        instance from the testing data and write result to file_to_write.
        """
        with open(filename_to_write, 'w') as f:
            f.write(f"id,class\n")  # First line 
            for i, X in enumerate(self.X_test):
                predict = self._predict(X)
                f.write(f"{i+1},{predict}\n")  # str to write
            

    
    def k_fold_cross_validation(self, k_folds):
        """
        Split the data into k equal sized folds, select one fold as test set and use the rest as 
        training set. Contiunue the process for k times until every fold has been selected as
        a test set once. Return the accuracy.
        
        Assuming k_folds is at least 2.
        """
        fold_size = int(len(self.X_train) / k_folds)
        total_count = fold_size * k_folds
        correct_count = 0

        # Run the classifier k_folds times on k_folds different test sets.
        for i in range(k_folds):
            # Split dataset into testing and training
            X_test = self.X_train[i*fold_size : (i+1)*fold_size]
            y_test = self.y_train[i*fold_size : (i+1)*fold_size]
            if i == 0:
                X_train = self.X_train[(i+1)*fold_size:]
                y_train = self.y_train[(i+1)*fold_size:]
            elif i == k_folds - 1:
                X_train = self.X_train[:i*fold_size]
                y_train = self.y_train[:i*fold_size]
            else:
                X_train = np.concatenate((self.X_train[:i*fold_size], self.X_train[(i+1)*fold_size:]))
                y_train = np.concatenate((self.y_train[:i*fold_size], self.y_train[(i+1)*fold_size:]))

            # Train the model on the selected training set
            self.train(X_train, y_train)

            # Run the learned model on the selected test set
            for j, x in enumerate(X_test):
                predict = self._predict(x)  # Predicted class of test data
                target = y_test[j]  # Target class of test data
                if predict == target:
                    correct_count += 1
        # Return the accuracy on the test set
        return correct_count / total_count

""" Load the data"""
""" Please expect a runnint time of about 30s"""

# Please expect about 30s running time to load data
model = NaiveBayes()
model.read("trg.csv", "tst.csv")

""" Finish loading the data from file """

# Train the model on the training set from input file
model.train()
# Run the model on the testing set from input file
# And write result to "solution.csv"
model.test("tzha842.csv")

# Run 5-fold cross validation on the training set from input file and get the accuracy of the model
k = 10
accuracy = model.k_fold_cross_validation(k)
print(f"Accuracy on {k}-fold cross validation: {100 * accuracy:5.2f}%")
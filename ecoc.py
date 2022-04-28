# Implementation of Error Correcting Output Codes
# See: https://proceedings.neurips.cc/paper/2019/file/cd61a580392a70389e27b0bc2b439f49-Paper.pdf
# Code at: https://github.com/Gunjan108/robust-ecoc

    #map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
    def encodeData(self):
        self.Y_train = np.zeros((self.data_dict['X_train'].shape[0], self.params_dict['M'].shape[1]))
        self.Y_test = np.zeros((self.data_dict['X_test'].shape[0], self.params_dict['M'].shape[1]))
        for k in np.arange(self.params_dict['M'].shape[1]):
            self.Y_train[:,k] = self.params_dict['M'][self.data_dict['Y_train_cat'], k]
            self.Y_test[:,k] = self.params_dict['M'][self.data_dict['Y_test_cat'], k]

    #this function takes the output of the NN and maps into logits (which will be passed into softmax to give a prob. dist.)
    #It effectively does a Hamming decoding by taking the inner product of the output with each column of the coding matrix (M)
    #obviously, the better the match, the larger the dot product is between the output and a given row
    #it is simply a log ReLU on the output
    def outputDecoder(self, x):
        
        mat1 = tf.matmul(x, self.params_dict['M'], transpose_b=True)
        mat1 = tf.log(tf.maximum(mat1, 0)+1e-6) #floor negative values
        return 
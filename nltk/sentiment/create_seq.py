from nltk.corpus import movie_reviews
from gensim.models import Word2Vec
from gensim import models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class reviews():

    def __init__(self, seq_len=2879):
        ## Train the word2vec
        #model = Word2Vec(movie_reviews.sents())
        print('Loading the model....')
        mr = models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('Word2vec.bin')
        
        print('Making a list of words of the review and its labels...')
        documents = [(list(movie_reviews.words(fieldid)), category)
                    for category in movie_reviews.categories()
                    for fieldid in movie_reviews.fileids(category)]
        '''
        For now only logic and rough draft
        '''
        ## Figuring out the longest sentence in the dataset
        sent_length = np.zeros((len(documents)))
        for i, tup in enumerate(documents):
            sent, r = tup
            sent_length[i] = len(sent)
        
        max_count = int(sent_length.max())

        self.seq_len = seq_len 
        self.input_size = mr.vector_size

        label_dic = {'pos':1, 'neg':0}
        
        X = np.zeros((len(documents), max_count, mr.vector_size)) # numpy arrays for the input
        y = np.zeros((len(documents))) # creating the labels
        
        count_counter = np.zeros((2000)) 
        for i, (sent, rev) in enumerate(documents):
            y[i] = label_dic[rev] # assigning the class label
            #one_review = [] # list for one review
            done = False
            sent_iter = iter(sent)
            count = 0
            while not done:
                try:
                    word = next(sent_iter)
                    try: # look for the word, and if found, add it on the X, and increment counter
                        w_vec = mr.get_vector(word)
                        X[i, count, :] = w_vec
                        count += 1
                    except:
                        pass
                except StopIteration: # no more words in the sentence
                    done = True
            count_counter[i] = count

            '''
            for j in range(max_count):
                if j < len(sent):
                    word = sent[j]
                    try:
                        w_vec = mr.wv.get_vector(word)
                        X[i, j, :] = w_vec
                    except:
                        pass
            '''
        # loading the arrays
        self.X = X[:, :self.seq_len, :]
        self.y = y

        # saving the arrays for quick access for next time
        #np.save('X.npy', self.X)
        #np.save('y.npy', self.y)
    @staticmethod
    def data_split(train_perc, valid_perc, X, y):
        # preprocess the perc
        train_perc = train_perc*0.01
        valid_perc = valid_perc*0.01
        test_perc  = 1 - (train_perc+valid_perc)
        valid_test_perc = 1 - train_perc
        ### Devide the data into train, validation, and test sets
        # pre set methods from sklearn for splitting datasets
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 
        # This can only split data into 2 sets, so there we'll create our own function
        # First, we'll split the data into training, and valid_test using sklearn

        m = X.shape[0] # num of all datapoints in X
        X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y.ravel(), test_size=valid_test_perc)

        # 2ndly, split the valid_test into valid and test set
        m_vt = X_valid_test.shape[0]
        valid_ind = int(m_vt*(valid_perc/valid_test_perc))

        X_valid = X_valid_test[:valid_ind, :, :]
        y_valid = y_valid_test[:valid_ind]

        X_test = X_valid_test[valid_ind:, :, :]
        y_test = y_valid_test[valid_ind:]

        return {'train_X':X_train, 'train_label':y_train, 'valid_X':X_valid, 'valid_label':y_valid, 'test_X':X_test, 'test_label':y_test}


    ## normalization for 3D 
    ## https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array
    @staticmethod
    def normalize(X):
        x_min = X.min(axis=(1, 2), keepdims=True)
        x_max = X.max(axis=(1, 2), keepdims=True)
        return (X - x_min)/(x_max - x_min)

    @staticmethod
    def pca(X, comp):
        pca = PCA(n_components=comp)
        result = np.zeros((X.shape[0], comp, X.shape[-1]))
        print('Applying PCA...')

        for row in range(X.shape[0]):
            result[row, :, :] = pca.fit_transform(X[row, :, :].T).T
        
        return result

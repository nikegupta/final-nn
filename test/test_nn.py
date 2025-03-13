# TODO: import dependencies and write unit tests below
from nn import nn, io, preprocess
import numpy as np
import random

def test_single_forward():
    neural_net = nn.NeuralNetwork([{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'}, {'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}],
                        0.1,1,2,3,'binary cross entropy',0.01)
    A_prev = np.array([[0.5,0.5,0.5],
                       [0.6,0.7,0.8]])
    W_curr = np.array([[0.5,0.5,0.5],
                       [0.6,0.7,0.8]])
    b_curr = np.array([[0.5],
                       [0.3]])
    activation = 'relu'

    result_A = np.array([[1.25,1.35],
                        [1.55,1.79]])
    result_Z = np.array([[1.25,1.35],
                        [1.55,1.79]])

    A_curr, Z_curr = neural_net._single_forward(W_curr,b_curr,A_prev,activation)
    for i in range(result_A.shape[0]):
        for j in range(result_Z.shape[1]):
            assert abs(A_curr[i][j] - result_A[i][j]) < 0.001
            assert abs(Z_curr[i][j] - result_Z[i][j]) < 0.001

    activation = 'sigmoid'
    result_A = np.array([[0.777299,0.794129],
                        [0.824913,0.856927]])
    
    A_curr, Z_curr = neural_net._single_forward(W_curr,b_curr,A_prev,activation)
    for i in range(result_A.shape[0]):
        for j in range(result_Z.shape[1]):
            assert abs(A_curr[i][j] - result_A[i][j]) < 0.001
            assert abs(Z_curr[i][j] - result_Z[i][j]) < 0.001

def test_forward():
    neural_net = nn.NeuralNetwork([{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'}, {'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    X_train = np.array([[0,0.1,0.2,0.3],
                        [0.8,0.9,1,1.1]])
    
    cache = neural_net.forward(X_train)
    assert cache['A1'][0][0] == 0.
    assert cache['A2'][0][1] == 0.
    assert abs(cache['Z2'][0][2] - 0.07832118) < 0.000001
    assert abs(cache['A3'][1][0] - 0.52779818) < 0.00001
    assert abs(cache['Z3'][1][1] - 0.08619111) < 0.000001

def test_single_backprop():
    neural_net = nn.NeuralNetwork([{'input_dim': 4, 'output_dim': 3, 'activation': 'relu'}, {'input_dim': 3, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,3,'binary cross entropy',0.01)
    A_prev = np.array([[0.5,0.5,0.5],
                       [0.6,0.7,0.8]])
    W_curr = np.array([[0.5,0.5,0.5],
                       [0.6,0.7,0.8]])
    b_curr = np.array([[0.5],
                       [0.3]])
    Z_curr = np.array([[1,6],
                       [9,3]])
    dA_curr = np.array([[10,11],
                        [3,7]])
    activation = 'relu'

    dA_prev, dW_curr, db_curr = neural_net._single_backprop(W_curr,b_curr,Z_curr,A_prev,dA_curr,activation)
    result_A = np.array([[44.6, 51.2, 57.8],
                        [26.1, 28.2, 30.3]])
    result_W = np.array([[10.6 , 11.95, 13.3],
                        [22.8 , 23.85, 24.9 ]])
    result_b = np.array([[18.5],
                        [43.5]])
    
    for i in range(result_A.shape[0]):
        for j in range(result_A.shape[1]):
            assert abs(dA_prev[i][j] - result_A[i][j]) < 0.001
            assert abs(dW_curr[i][j] - result_W[i][j]) < 0.001

    for i in range(result_b.shape[0]):
        for j in range(result_b.shape[1]):
            assert abs(db_curr[i][j] - result_b[i][j]) < 0.001
    

def test_predict():
    neural_net = nn.NeuralNetwork([{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    X_train = np.array([[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                    [0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],
                    [1.6,1.7,1.8,1.9,2,2.1,2.3,2.4],
                    [2.5,2.6,2.7,2.8,2.9,3,3.1,3.2]])
    result = np.array([[0.50313357,0.48324257],
                    [0.50652897,0.48171587],
                    [0.51050615,0.47992765],
                    [0.5137418,0.47847278]])
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            assert abs(neural_net.predict(X_train)[i][j] - result[i][j]) < 0.0001
    


def test_binary_cross_entropy():
    neural_net = nn.NeuralNetwork([{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    
    y = np.array([[0,1],
                  [1,0]])
    y_hat = np.array([[0.1,0.8],
                      [0.9,0.3]])
    
    assert abs(neural_net._binary_cross_entropy(y,y_hat) - 0.17166) < 0.0001

def test_binary_cross_entropy_backprop():
    neural_net = nn.NeuralNetwork([{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    
    y = np.array([[0,1],
                  [1,0]])
    y_hat = np.array([[0.1,0.8],
                      [0.9,0.3]])
    
    result = np.array([[1.11111111,-1.25],
                        [-1.11111111,1.42857143]])
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            assert abs(neural_net._binary_cross_entropy_backprop(y,y_hat)[i][j] - result[i][j]) < 0.0001

def test_mean_squared_error():
    neural_net = nn.NeuralNetwork([{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    
    y = np.array([[0,1],
                  [1,0]])
    y_hat = np.array([[0.1,0.8],
                      [0.9,0.3]])
    
    assert abs(neural_net._mean_squared_error(y,y_hat) - 0.075) < 0.0001

def test_mean_squared_error_backprop():
    neural_net = nn.NeuralNetwork([{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}],
                          0.1,1,2,2,'binary cross entropy',0.01)
    
    y = np.array([[0,1],
                  [1,0]])
    y_hat = np.array([[0.1,0.8],
                      [0.9,0.3]])
    
    result = np.array([[0.1,-0.2],
                        [-0.1,0.3]])
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            assert abs(neural_net._mean_squared_error_backprop(y,y_hat)[i][j] - result[i][j]) < 0.0001

def test_sample_seqs():
    random.seed(1)
    positive_seqs = io.read_text_file('data/rap1-lieb-positives.txt')[:2]
    postive_labels = [1 for x in range(len(positive_seqs))]

    negative_seqs = io.read_fasta_file('data/yeast-upstream-1k-negative.fa')
    negative_seqs_truncated = []
    for i in range(len(negative_seqs)):
        start_index = random.randint(0,len(negative_seqs[0]) - len(positive_seqs[0]))
        negative_seqs_truncated.append(negative_seqs[0][start_index:start_index + len(positive_seqs[0])])
    negative_labels = [0 for x in range(len(negative_seqs_truncated))]
    seqs, labels = preprocess.sample_seqs(positive_seqs + negative_seqs_truncated, postive_labels + negative_labels)
    encodings = preprocess.one_hot_encode_seqs(seqs)
    X = np.array(encodings)
    assert X.shape == (4,68)

def test_one_hot_encode_seqs():
    assert preprocess.one_hot_encode_seqs('AGA') == [[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
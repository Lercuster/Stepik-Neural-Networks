import numpy as np

class Perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def forward_pass(self, single_input):
        
        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i]*single_input[i]
        result += self.b
        if result > 0:
            return 1
        else:
            return 0
    
    def vectorized_forward_pass(self, input_matrix):
        """ input_matrix - матрица (n, m) срока - пример, м - кол-во переменных.
            self.w - веса, размера (m, 1)
            b - число, смещение
            result - вектор (n, 1)
            """
        result = np.array(float)
        
        result = input_matrix.dot(self.w)
        result = result.reshape(len(result), 1)
        #print(result.shape)
        result += self.b*np.ones(1)
        is_positive = result > 0
        return is_positive
        
    def train_on_single_example(self, example, y):
        """ error - значения на вход
            predict - предсказание
            y - известные ответы
            """
        predict = (self.w.T.dot(example) + self.b) > 0
        error = y - predict
        self.w = self.w + error*example
        self.b = self.b + error
        return error
    
    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        """
        input_matrix - матрица входов размера (n, m),
        y - вектор правильных ответов размера (n, 1) (y[i] - правильный ответ на пример input_matrix[i]),
        max_steps - максимальное количество шагов.
        Применяем train_on_single_example, пока не перестанем ошибаться или до умопомрачения.
        Константа max_steps - наше понимание того, что считать умопомрачением.
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)  

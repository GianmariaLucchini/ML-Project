import numpy as np
import cvxopt

class my_SVM_train(object):

    def __init__(self, kernel_type, kernel):
        self.kernel_type = kernel_type
        self.kernel = kernel

    def train(self, X, y):

        lagrange_multipliers = self.compute_lagrange(X, y)

        return self.predictor(X, y, lagrange_multipliers)

    def predictor(self, X, y, lagrange_multipliers):
        index = lagrange_multipliers > 1e-5

        weights = lagrange_multipliers[index]
        support_vectors = X[index]
        support_vector_y = y[index]

        average = []
        for (y_k, x_k) in zip(support_vector_y, support_vectors):
            average.append(y_k - my_SVM_pred(self.kernel_type, self.kernel, 0.0, weights, support_vectors, support_vector_y).predict(x_k))

        bias = np.mean(average)

        return my_SVM_pred(self.kernel_type, self.kernel, bias, weights, support_vectors, support_vector_y)


    def compute_lagrange(self, X, y):
        samples, features = X.shape

        # Compute Gram Matrix

        K = np.zeros((samples, samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)

        # Solves the Given QP ( through cvxopt )

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(samples))

        _G = cvxopt.matrix(np.diag(np.ones(samples) * -1))
        _h = cvxopt.matrix(np.zeros(samples))

        G_ = cvxopt.matrix(np.diag(np.ones(samples)))
        h_ = cvxopt.matrix(np.ones(samples))

        G = cvxopt.matrix(np.vstack((_G, G_)))
        h = cvxopt.matrix(np.vstack((_h, h_)))

        my_cvxopt = cvxopt.solvers
        my_cvxopt.options['show_progress'] = False
        solution = my_cvxopt.qp(P, q, G, h)

        # Lagrange multipliers

        return np.ravel(solution['x'])


class my_SVM_pred(object):

    def __init__(self, kernel_type, kernel, bias, weights, support_vectors, support_vector_y):
        self.kernel_type = kernel_type
        self.kernel = kernel
        self.bias = bias
        self.weights = weights
        self.support_vectors = support_vectors
        self.support_vector_y = support_vector_y

    def predict(self, x):
        result = self.bias

        if self.kernel_type in "linear":

            for z_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vector_y):
                result += z_i * y_i * self.kernel(x_i, x)

            return np.sign(result)

        else:

            results = []
            for z_i, x_i, y_i in zip(self.weights, self.support_vectors, self.support_vector_y):
                result += z_i * y_i * self.kernel(x_i, x)
                if np.sign(result) > 0:
                    results.append(1)
                else:
                    results.append(-1)
            return results

    def score(self, y1, y2):
        size, aux = y2.shape
        ok = 0
        for k in range(0,size):
            if y1[k] == y2[k]:
                ok += 1
        return ok/size


class my_Kernel(object):

    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

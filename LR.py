import random
import numpy as np
import matplotlib.pyplot as plt
import time


class LogisticRegression:
    def __init__(self, num_iters, alpha,with_reg=False, with_monentum=False):
        self.num_iters = num_iters
        self.alpha = alpha
        self.with_reg = with_reg
        self.with_monentum = with_monentum
        self.reg_strength = 1e-3
        self.momentum = 0.9
        self.velocity = None
        self.w = None
        self.b = None

    def train(self,train_data,train_label, Y, test_data, test_label, print_loss):
        losses = []
        trainAcc = []
        testAcc = []
        t0 = time.time()
        self.w, self.b = self.init_weight(train_data.shape[0], train_label.shape[0])
        self.velocity = np.zeros(self.w.shape)
        for i in range(self.num_iters):
            grads, loss = self.cal_grads_and_loss(self.w, self.b, train_data, train_label)
            parameters = self.optimizer(grads)
            trainAcc.append(self.cal_accur(train_data, Y))
            testAcc.append(self.cal_accur(test_data, test_label))
            # if i % 50 == 0:
            losses.append(loss)
            # Print the cost every 100 training examples
            if print_loss and i % 50 == 0:
                print("Cost after iteration %i: %f" % (i, loss))
        w = parameters["w"]
        b = parameters["b"]
        label_prediction_train = self.predict(w, b, train_data)
        label_prediction_test = self.predict(w, b, test_data)
        print('Train accuracy: %.3f %%' % (sum(label_prediction_train == Y) / (float(len(Y))) * 100))
        print('Test accuracy: %.3f %%' % (sum(label_prediction_test == test_label) / (float(len(test_label))) * 100))
        print('Finished Training，take %.3f s' % (time.time() - t0))

        results = {"losses": losses,
              "label_prediction_test": self.predict(w, b, test_data),
              "label_prediction_train": self.predict(w, b, train_data),
              "w": w,
              "b": b,
              "learning_rate": self.alpha,
              "num_iterations": self.num_iters}

        self.plotGraph(losses,trainAcc,testAcc)

        return results

    def optimizer(self,grads):
        dw = grads["dw"]
        db = grads["db"]
        if self.with_monentum == True:
            self.velocity = (self.momentum * self.velocity) + (self.alpha * dw)
            self.w -= self.velocity
        else:
            self.w = self.w - self.alpha * dw
        self.b = self.b - self.alpha * db
        self.alpha = self.alpha * 0.99
        params = {"w": self.w, "b": self.b}
        return params

    def softmax_func(self, score):
        score -= np.max(score)
        final_score = (np.exp(score).T / np.sum(np.exp(score), axis=1))
        return final_score

    def init_weight(slef, dim1, dim2):
        # w = np.ones(shape=(dim1, dim2))
        w = 0.001 * np.random.rand(dim1, dim2)
        # w = np.random.randn(dim1, dim2) / np.sqrt(dim1 / 2)
        b = np.zeros(shape=(10, 1))
        return w, b


    def cal_grads_and_loss(self, w, b, data,label):
        # getting no of rows
        epsilon = 1e-5
        m = data.shape[1]
        # score = (np.dot(w.T, data) + b).T
        final_score = self.softmax_func((np.dot(w.T, data) + b).T)
        loss = (-1 / m) * np.sum(label * np.log(final_score+epsilon))
        if self.with_reg:
            loss += 0.5 * self.reg_strength * (w ** 2).sum()
        # backwar prop
        if self.with_reg:
            dw = (1 / m) * np.dot(data, (final_score - label).T) + (self.reg_strength * w)
        else:
            dw = (1 / m) * np.dot(data, (final_score - label).T)
        db = (1 / m) * np.sum(final_score - label)

        loss = np.squeeze(loss)
        grads = {"dw": dw,
                 "db": db}
        return grads, loss

    def cal_accur(self, data, label):
        prediction = self.predict(self.w,self.b,data)
        return (sum(prediction == label) / (float(len(label))) * 100)

    def sample(self, X, Y, batch_size, w, b):
        random_indices = random.sample(range(X.shape[1]), batch_size)
        X_batch = X.T[random_indices].T
        Y_batch = Y.T[random_indices].T
        return X_batch, Y_batch

    def predict(self, w, b, X):
        label_pred = np.argmax(self.softmax_func((np.dot(w.T, X) + b).T), axis=0)
        return label_pred


    def plotGraph(self, trainLosses, trainAcc, testAcc):
        plt.subplot(1,2, 1)
        plt.plot(trainLosses, label="Train loss")
        # plt.plot(testLosses, label="Test loss")
        plt.legend(loc='best')
        plt.title("Epochs vs. Cross Entropy Loss")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Cross Entropy Loss")
        # plt.show()

        plt.subplot(1,2, 2)
        plt.plot(trainAcc, label="Train Accuracy")
        plt.plot(testAcc, label="Test Accuracy")
        plt.legend(loc='best')
        plt.title("Epochs vs. Mean per class Accuracy")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Mean per class Accuracy")
        plt.tight_layout(pad=0.5, w_pad=1, h_pad=1)

        plt.show()
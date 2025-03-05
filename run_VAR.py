import pandas as pd
import numpy as np
import os
from statsmodels.tsa.api import VAR
from scipy.linalg import inv
import argparse
import yaml
from metircs.metrics import RMSE_MAE_MAPE


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def preprocess_data(data, in_steps=3, out_steps=1, test_size=0.2, val_size=0.2):
    '''
    :return: X is [B, in_steps, ...], Y is [B, out_steps, ...]
    '''
    length = len(data)
    end_index = length - in_steps - out_steps + 1
    X = []     # in
    Y = []     # out
    index = 0

    while index < end_index:
        X.append(data[index:index+in_steps])
        Y.append(data[index+in_steps:index+in_steps+out_steps])
        index = index+1
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    data_len = X.shape[0]
    train_input = X[:-int(data_len*(test_size))]
    train_output = Y[:-int(data_len*(test_size))]
    test_input = X[-int(data_len*test_size):]
    test_output = Y[-int(data_len*test_size):]
    return train_input, train_output, test_input, test_output

def get_data(dataset):
    # path
    if dataset in {'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'}:
        data_path = os.path.join(".", "data", dataset, dataset + '.npz')
        data = np.load(data_path)['data'][:, :, :1]
    elif dataset in {'METR-LA', 'PEMS-BAY'}:
        data_path = os.path.join(".", "data", dataset, dataset + '.h5')
        data = pd.read_hdf(data_path).values
        data = data[:, :, np.newaxis]
    else:
        raise ValueError

    return data


def run_VAR(data, inputs):
    ts, points, f = data.shape
    train_rate = 1-args.test_size-args.val_size
    eval_rate = args.val_size
    output_window = args.out_steps
    maxlags = args.model_args['maxlags']

    data = data.reshape(ts, -1)[:int(ts * (train_rate + eval_rate))]  # (train_size, N * F)
    scaler = StandardScaler(data.mean(), data.std())
    data = scaler.transform(data)

    model = VAR(data)
    try:
        results = model.fit(maxlags=maxlags, ic='aic')
    except np.linalg.LinAlgError:
        print("遇到非正定矩阵问题，尝试添加正则化项...")
        # 添加正则化项
        reg_param = 1e-6
        nobs = data.shape[0]
        nvar = data.shape[1]
        Y = data[maxlags:]
        Z = np.hstack([data[t:t + maxlags].flatten() for t in range(nobs - maxlags)])
        Z = Z.reshape(-1, nvar * maxlags)
        ZZ = np.dot(Z.T, Z) + reg_param * np.eye(Z.shape[1])
        ZY = np.dot(Z.T, Y)
        coefs = np.dot(inv(ZZ), ZY)
        residuals = Y - np.dot(Z, coefs)
        sigma_u = np.dot(residuals.T, residuals) / (nobs - maxlags - nvar * maxlags)
        class DummyResults:
            def __init__(self, coefs, sigma_u):
                self.coefs = coefs
                self.sigma_u = sigma_u

            def forecast(self, y, steps):
                nobs = y.shape[0]
                nvar = y.shape[1]
                result = np.zeros((steps, nvar))
                for i in range(steps):
                    y_lagged = y[-maxlags:].flatten()
                    result[i] = np.dot(coefs.T, y_lagged)
                    y = np.vstack([y[1:], result[i]])
                return result

        results = DummyResults(coefs, sigma_u)

    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)  # (num_samples, out, N * F)
    y_pred = []  # (num_samples, out, N, F)
    for sample in inputs:  # (out, N * F)
        sample = scaler.transform(sample[-maxlags:])  # (T, N, F)
        out = results.forecast(sample, output_window)  # (out, N * F)
        out = scaler.inverse_transform(out)  # (out, N * F)
        y_pred.append(out.reshape(output_window, points, f))
    y_pred = np.array(y_pred)  # (num_samples, out, N, F)
    return y_pred


def main(args):
    data = get_data(args.dataset)
    trainx, trainy, testx, testy = preprocess_data(data, args.in_steps, args.out_steps, args.test_size, args.val_size)
    y_pred = run_VAR(data, testx)
    y_pred = y_pred[:, :, :, 0]
    y_true = testy[:, :, :, 0]
    out_steps = y_pred.shape[1]

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (rmse_all, mae_all, mape_all,)
    # test metric
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (i + 1, rmse, mae, mape,)
    print(out_str)

    log_file_path = "log/" + 'VAR' + '_' + args.dataset + '.log'
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_file_path, 'w') as log_file:
        log_file.write(out_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems-bay")
    args = parser.parse_args()
    args.dataset = args.dataset.upper()
    with open(f"./configs/VAR.yaml", "r") as f:
        args.__dict__.update(yaml.safe_load(f)[args.dataset])

    main(args)

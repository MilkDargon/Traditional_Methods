from tqdm import tqdm
from sklearn.svm import SVR

import yaml
import numpy as np
import argparse
import os
import pandas as pd
from metircs.metrics import RMSE_MAE_MAPE
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


def run_SVR(data):
    ts, num_nodes, f = data.shape
    output_window = args.out_steps
    kernel = args.model_args['kernel']
    y_pred = []
    y_true = []
    # 使用 tqdm 创建进度条
    with tqdm(range(num_nodes), desc='num_nodes') as pbar:
        for i in pbar:
            trainx, trainy, testx, testy = preprocess_data(data[:, i, :], args.in_steps, args.out_steps, args.test_size, args.val_size)  # (T, F)
            # (train_size, in/out, F), (test_size, in/out, F)
            trainx = np.reshape(trainx, (trainx.shape[0], -1))  # (train_size, in * F)
            trainy = np.reshape(trainy, (trainy.shape[0], -1))  # (train_size, out * F)
            trainy = np.mean(trainy, axis=1)  # (train_size,)
            testx = np.reshape(testx, (testx.shape[0], -1))  # (test_size, in * F)
            # 避免使用 print 打断进度条，可将信息添加到进度条后缀
            shape_info = f"trainx: {trainx.shape}, trainy: {trainy.shape}, testx: {testx.shape}, testy: {testy.shape}"
            pbar.set_postfix_str(shape_info)
            svr_model = SVR(kernel=kernel)
            svr_model.fit(trainx, trainy)
            pre = svr_model.predict(testx)  # (test_size, )
            pre = np.expand_dims(pre, axis=1)  # (test_size, 1)
            pre = pre.repeat(output_window * f, axis=1)  # (test_size, out * F)
            y_pred.append(pre.reshape(pre.shape[0], output_window, f))
            y_true.append(testy)
    y_pred = np.array(y_pred)  # (N, test_size, out, F)
    y_true = np.array(y_true)  # (N, test_size, out, F)
    y_pred = y_pred.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    y_true = y_true.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    return y_pred, y_true


def main(args):
    data = get_data(args.dataset)
    y_pred, y_true = run_SVR(data)
    y_pred = y_pred[:, :, :, 0]
    y_true = y_true[:, :, :, 0]
    out_steps = y_pred.shape[1]

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (rmse_all, mae_all, mape_all,)
    # test metric
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (i + 1, rmse, mae, mape,)
    print(out_str)

    log_file_path = "log/"+'SVR'+'_'+args.dataset+'.log'
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_file_path, 'w') as log_file:
        log_file.write(out_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems04")
    args = parser.parse_args()
    args.dataset = args.dataset.upper()
    with open(f"./configs/SVR.yaml", "r") as f:
        args.__dict__.update(yaml.safe_load(f)[args.dataset])

    main(args)

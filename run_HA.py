import yaml
import numpy as np
import argparse
import os
import pandas as pd
from metircs.metrics import RMSE_MAE_MAPE

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


def historical_average(data):
    t, n, f = data.shape
    train_rate = 1-args.test_size-args.val_size
    eval_rate = args.val_size
    output_window = args.out_steps
    weight = args.model_args["weight"]
    null_value = args.model_args["null_value"]

    parts = args.lag.split('*')
    lag = 1
    for part in parts:
        lag *= int(part)

    if isinstance(lag, int):
        lag = [lag]
    if isinstance(weight, int) or isinstance(weight, float):
        weight = [weight]
    assert sum(weight) == 1

    y_true = []
    y_pred = []
    for i in range(int(t * (train_rate + eval_rate)), t):
        # y_true
        y_true.append(data[i, :, :])  # (N, F)
        # y_pred
        y_pred_i = 0
        for j in range(len(lag)):
            # 隔lag[j]时间步在整个训练集采样, 得到(n_sample, N, F)取平均值得到(N, F), 最后用weight[j]加权
            inds = [j for j in range(i % lag[j], int(t * (train_rate + eval_rate)), lag[j])]
            history = data[inds, :, :]
            # 对得到的history数据去除空值后求平均
            null_mask = (history == null_value)
            history[null_mask] = np.nan
            y_pred_i += weight[j] * np.nanmean(history, axis=0)
            y_pred_i[np.isnan(y_pred_i)] = 0
        y_pred.append(y_pred_i)  # (N, F)

    y_pred = np.array(y_pred)  # (test_size, N, F)
    y_true = np.array(y_true)  # (test_size, N, F)
    y_pred = np.expand_dims(y_pred, axis=1)  # (test_size, 1, N, F)
    y_true = np.expand_dims(y_true, axis=1)  # (test_size, 1, N, F)
    y_pred = np.repeat(y_pred, output_window, axis=1)  # (test_size, out, N, F)
    y_true = np.repeat(y_true, output_window, axis=1)  # (test_size, out, N, F)
    return y_pred, y_true


def main(args):
    data = get_data(args.dataset)
    y_pred, y_true = historical_average(data)
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

    log_file_path = "log/"+'HA'+'_'+args.dataset+'.log'
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_file_path, 'w') as log_file:
        log_file.write(out_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="metr-la")
    args = parser.parse_args()
    args.dataset = args.dataset.upper()
    with open(f"./configs/HA.yaml", "r") as f:
        args.__dict__.update(yaml.safe_load(f)[args.dataset])


    main(args)

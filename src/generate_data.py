# -*- coding: utf-8 -*-
import os
import requests
import json
import time
import random
import itertools
import numpy as np
from tqdm import tqdm
from preprocessing import (
    read_ts_dataset,
    normalize_dataset,
    moving_windows_preprocessing,
    denormalize,
)

NUM_CORES = 7


def notify_slack(msg, webhook=None):
    if webhook is None:
        webhook = os.environ.get("webhook_slack")
    if webhook is not None:
        try:
            requests.post(webhook, json.dumps({"text": msg}))
        except:
            print("Error while notifying slack")
            print(msg)
    else:
        print("NO WEBHOOK FOUND")



# Preprocessing parameters
with open("parameters_reduced_demand_168.json") as f:
    PARAMETERS = json.load(f)

# PARAMETERS = json.load("parameters.json")
NORMALIZATION_METHOD = PARAMETERS["normalization_method"]
PAST_HISTORY_FACTOR = PARAMETERS["past_history_factor"]
FORECAST_HORIZON = PARAMETERS["forecast_horizon"]

# This variable stores the urls of each dataset.
# DATASETS = json.load("../data/datasets.json")


DATASET_NAMES = [name for name in os.listdir("../data")]


def generate_dataset(args):
    dataset, norm_method, past_history_factor, forecast_horizon = args


    train = read_ts_dataset("../data/{}/train.csv".format(dataset))
    test = read_ts_dataset("../data/{}/test.csv".format(dataset))
    forecast_horizon = forecast_horizon #24  # test.shape[1]

    print(
        dataset,
        {
            "Max length": np.max([ts.shape[0] for ts in train]),
            "Min length": np.min([ts.shape[0] for ts in train]),
            "Forecast Horizon": forecast_horizon,
        },
        )

    # Normalize data
    train, test, norm_params = normalize_dataset(
        train, test, norm_method, dtype="float32"
    )



    norm_params_json = [{k: float(p[k]) for k in p} for p in norm_params]
    norm_params_json = json.dumps(norm_params_json)

    if not os.path.exists('../data/{}/{}/'.format(dataset, norm_method)):
        os.mkdir('../data/{}/{}/'.format(dataset, norm_method))

    with open("../data/{}/{}/norm_params.json".format(dataset, norm_method), "w") as file:
        file.write(norm_params_json)


    # Format training and test input/output data using the moving window strategy

    past_history = int(forecast_horizon * past_history_factor)

    x_train, y_train, x_test, y_test = moving_windows_preprocessing(
        train, test, past_history, forecast_horizon, np.float32, n_cores=1
    )

    y_test_denorm = np.copy(y_test)

    for i in range(y_test.shape[0]):
        y_test_denorm[i] = denormalize(y_test[i], norm_params[0], method=norm_method)

    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print()
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)

    print(f'PAST HISTORY FACTOR -> {past_history_factor}')
    if not os.path.exists('../data/{}/{}/{}/'.format(dataset, norm_method, past_history_factor)):
        os.mkdir('../data/{}/{}/{}/'.format(dataset, norm_method, past_history_factor))

    if not os.path.exists('../data/{}/{}/{}/{}'.format(dataset, norm_method, past_history_factor, forecast_horizon)):
        os.mkdir('../data/{}/{}/{}/{}'.format(dataset, norm_method, past_history_factor, forecast_horizon))

    np.save(
        "../data/{}/{}/{}/{}/x_train.np".format(dataset, norm_method, past_history_factor, forecast_horizon),
        x_train,
    )
    np.save(
        "../data/{}/{}/{}/{}/y_train.np".format(dataset, norm_method, past_history_factor, forecast_horizon),
        y_train,
    )
    np.save(
        "../data/{}/{}/{}/{}/x_test.np".format(dataset, norm_method, past_history_factor, forecast_horizon),
        x_test,
    )
    np.save(
        "../data/{}/{}/{}/{}/y_test.np".format(dataset, norm_method, past_history_factor, forecast_horizon),
        y_test,
    )
    np.save(
        "../data/{}/{}/{}/{}/y_test_denorm.np".format(
            dataset, norm_method, past_history_factor, forecast_horizon
        ),
        y_test_denorm,
    )


params = [
    (dataset, norm_method, past_history_factor, forecast_horizon)
    for dataset, norm_method, past_history_factor, forecast_horizon in itertools.product(
        DATASET_NAMES, NORMALIZATION_METHOD, PAST_HISTORY_FACTOR, FORECAST_HORIZON
    )
]


for i, args in tqdm(enumerate(params)):
    t0 = time.time()
    generate_dataset(args)
    dataset, norm_method, past_history_factor, forecast_horizon = args
    notify_slack(
        "[{}/{}] Generated dataset {} with {} normalization, past history factor of {} and forecast horizon of {} ({:.2f} s)".format(
            i, len(params), dataset, norm_method, past_history_factor, forecast_horizon, time.time() - t0
        )
    )
    print(
        "[{}/{}] Generated dataset {} with {} normalization, past history factor of {} and forecast horizon of {} ({:.2f} s)".format(
            i, len(params), dataset, norm_method, past_history_factor, forecast_horizon, time.time() - t0
        )
    ) 
    

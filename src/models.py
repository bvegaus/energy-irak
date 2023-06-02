import tensorflow as tf
from tensorflow_addons.layers import ESN
from tcn import TCN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression


def mlp(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        hidden_layers=[32, 16, 8],
        dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model




def lstm(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.LSTM(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.LSTM(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def tcn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16],
        tcn_dropout=0.0,
        return_sequences=True,
        activation="linear",
        padding="causal",
        use_skip_connections=True,
        use_batch_norm=False,
        dense_layers=[],
        dense_dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        use_skip_connections=use_skip_connections,
        dropout_rate=tcn_dropout,
        activation=activation,
        use_batch_norm=use_batch_norm,
        padding=padding,
    )(inputs)
    # Dense block
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model



def xgb(params):
    booster, n_estimators, min_child_weight, subsample, colsample_bytree, max_depth = params
    model = XGBRegressor(booster=booster, colsample_bytree=colsample_bytree, 
                         max_depth=max_depth, min_child_weight=min_child_weight, n_estimators=n_estimators,
                         n_jobs=-1, subsample=subsample)
    return model
    

def lr(params):
    fit_intercept, normalize, positive = params
    model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=-1, positive=positive)
    return model

def rf(params):

    n_stimators_value, max_depth_value, min_samples_split_value, min_samples_leaf_value = params

    model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=n_stimators_value,
                                  max_depth=max_depth_value, min_samples_split=min_samples_split_value,
                                  min_samples_leaf=min_samples_leaf_value)
    return model




def create_rnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(
        input_shape=input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss="mae",
        recurrent_units=[args["units"]] * args["layers"],
        return_sequences=args["return_sequence"],
    )





model_factory = {
    "mlp": mlp,
    "lstm": create_rnn(lstm),
    "tcn": tcn,
}

model_factory_ML = {
    "xgb": xgb,
    "lr":lr,
    "rf": rf
    
}


def create_model(model_name, input_shape, **args):
    assert model_name in model_factory.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory[model_name](input_shape, **args)


def create_model_ml(model_name, params):
    assert model_name in model_factory_ML.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory_ML[model_name](params)

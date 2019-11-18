
import keras
import pandas as pd
from keras import Sequential
from keras.layers import Dense

# region SAVE - LOAD - CREATE
def save_model(model, path):
    model.save_weights(path + ".h5", overwrite=True)
    with open(path + ".json", "w") as outfile:
        json.dump(model.to_json(), outfile)

def load_model(s_alg, path):
    optimizer_config = []
    print(path)

    if s_alg == "DSRL":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type_near":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type_near_propNeg":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_object_near":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_object":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0, 1])


        use_optimizer, optimizer_config = define_optimizer(optimizer)
        model.compile(loss='mse', optimizer=use_optimizer)
        model.summary()
        # pass
    return model, optimizer_config

def  create_model(s_alg, state_shape, net_conf):
    optimizer_config = []
    m_index = pd.MultiIndex(levels=[[''], [""]],
                                labels=[[], []],
                                names=['state', 'actions'])
    model = pd.DataFrame(index=m_index)

    return model, optimizer_config

# endregion

def define_optimizer(s_optimizer):
    lr = 0
    beta_1 = 0
    beta_2 = 0
    epsilon = 0
    decay = 0
    rho = 0
    if s_optimizer == "adam":
        lr = 0.001  # 0.001
        beta_1 = 0.9  # 0.9
        beta_2 = 0.999  # 0.999
        epsilon = 1e-08  # 1e-08
        decay = 0.0  # 0.0
        optimizer_selected = keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    elif s_optimizer == "rms_opt":
        lr = 0.001  # 0.001
        rho = 0.9  # 0.9
        epsilon = 1e-08  # e-08
        decay = 0.0  # 0.0
        optimizer_selected = keras.optimizers.RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
    optimizer_config = [lr, beta_1, beta_2, epsilon, decay, rho]
    return optimizer_selected, optimizer_config
#

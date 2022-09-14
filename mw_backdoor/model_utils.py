import lightgbm as lgb
def train_model(model_id, x_train, y_train):
    """ Train an EmberNN classifier

    :param model_id: (str) model type
    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained classifier
    """

    if model_id == 'lightgbm':
        return train_lightgbm(
            x_train=x_train,
            y_train=y_train
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))

def train_lightgbm(x_train, y_train):
    """ Train a LightGBM classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained LightGBM classifier
    """

    lgbm_dataset = lgb.Dataset(x_train, y_train)
    lgbm_model = lgb.train({"application": "binary",'verbose':-1}, lgbm_dataset)

    return lgbm_model

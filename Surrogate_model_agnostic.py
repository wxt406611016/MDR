from mw_backdoor.defense_tool import Defense_3
import tensorflow as tf
import argparse
import pandas as pd
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--target',type=str,default='combined')
args = parser.parse_args()
Contagio = 6000
strategy = args.target
wm_size = [16]
surrogate_model = ['linearsvc','RF','DNN']
surrogate_model_pos = {
    'linearsvc':None,
    'RF': './RF_backdoored.pickle',
    'DNN':'./DNN_backdoored.h5'
}
poison_rate = [0.08] if strategy == 'combined' else [0.02]
for size in wm_size:
    for p_r in poison_rate:
        for s_m in surrogate_model:
            backdoor_model_pos = f'./backdoor_pdf_{size}/{strategy}/{surrogate_model_pos[s_m]}' if surrogate_model_pos[s_m] else None
            X_train_pos = f'./backdoor_pdf_{size}/{strategy}/{int(Contagio*p_r)}_watermarked_X.npy'
            y_train_pos = f'./backdoor_pdf_{size}/{strategy}/{int(Contagio*p_r)}_watermarked_y.npy'
            config_pos = f'./backdoor_pdf_{size}/{strategy}/{int(Contagio*p_r)}_wm_config.npy'
            defense = Defense_3(p_r,backdoor_model_pos,X_train_pos,y_train_pos,None,strategy,config_pos,dataset='contagio',surrogate_model=s_m,wm_size=size)
            defense.MDR()
            print(defense.show())
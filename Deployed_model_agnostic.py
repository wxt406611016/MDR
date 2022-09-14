from mw_backdoor.transfer_3 import Transfer
import tensorflow as tf
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--target',type=str,default='combined')
args = parser.parse_args()
strategy = args.target
poison_rate = 0.08 if strategy == 'combined' else 0.02
Contagio = 6000
poison_num = int(Contagio*poison_rate)
wm_size = 16
x_test_pos = f'./backdoor_pdf_16/{strategy}/{int(Contagio*poison_rate)}_watermarked_X_test.npy'
x_train_pos = f'./backdoor_pdf_16/{strategy}/{int(Contagio*poison_rate)}_watermarked_X.npy'
y_train_pos = f'./backdoor_pdf_16/{strategy}/{int(Contagio*poison_rate)}_watermarked_y.npy'
new_x_pos = f'./backdoor_pdf_16/{strategy}/MDR_X_linearsvc.npy'
new_y_pos = f'./backdoor_pdf_16/{strategy}/MDR_y_linearsvc.npy'
transfer = Transfer(x_test_pos,x_train_pos,y_train_pos,new_x_pos,new_y_pos,poison_num,wm_size,strategy)
transfer.run()
print(transfer.overview())
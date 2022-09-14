"""
@author: supervampire
@email: wangxutong@iie.ac.cn
@version: Created in 2022 03/25.
"""
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import pandas as pd
from mw_backdoor import attack_utils
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
import os
import time
from mw_backdoor import config
from mw_backdoor import model_utils
from mw_backdoor import defense_utils
from sklearn.ensemble import IsolationForest
import community as community_louvain
from mw_backdoor.transfer_3 import Deep_NN
import shap
from mw_backdoor.wm_identify_tool import wm_identify

class Defense_3:
    def __init__(self,poison_rate,backdoor_model_pos,X_train_pos,y_train_pos,X_test_pos,strategy,config_pos,dataset,surrogate_model,wm_size):
        """
        @ params:
            posion_num : the number of benign samples with watermark
            backdoor_model_pos : the position of trained backdoor_model
            X_train_pos : the position of X_train dataset
            X_test_pos : the position of X_test dataset
            strategy : Attack Method
            summary : summary of defense result
        """
        self.wm_size = wm_size
        self.X_train_pos = X_train_pos
        self.poison_rate = poison_rate
        self.X_train = np.load(X_train_pos)
        self.poison_num = int(self.X_train.shape[0] * self.poison_rate)
        self.y_train = np.load(y_train_pos)
        self.middle = int(self.X_train.shape[0]/2)
        self.length = self.X_train.shape[1]
        if self.wm_size != 0:
            self.config = self.config_show(config_pos)
        self.strategy = strategy
        self.dataset = dataset
        self.surrogate_model = surrogate_model
        self.summary(version = 1)
        self.wm_i = None
        self.wm_i_c = None
        if self.dataset == 'contagio':
            if backdoor_model_pos:
                print(f'[+] Proxy_model found, Using the found proxy model')
                if self.surrogate_model == 'RF':
                    file = open(backdoor_model_pos,'rb')
                    self.backdoor_model = pickle.load(file)
                    file.close()
                elif self.surrogate_model == 'lightgbm':
                    self.backdoor_model = lgb.Booster(model_file = backdoor_model_pos)
                else:
                    self.backdoor_model = Deep_NN(135)
                    self.backdoor_model.model.load_weights(backdoor_model_pos)
                    self.backdoor_model.normal.fit(self.X_train)
            else:
                self.backdoor_model = self.build_model()
        else:
            self.backdoor_model = lgb.Booster(model_file = backdoor_model_pos) if backdoor_model_pos else self.build_model()

        if self.dataset == 'ember': 
            self.ori_model = lgb.Booster(model_file = './dataset/ember_lightgbm')
            self.X_test = np.load(X_test_pos)
            self.y_ori_label = np.load('./dataset/y_test.npy')
            self.x_ori = np.load('./dataset/x_test.npy')
            self.x_mw_poisoning_candidates, self.x_mw_poisoning_candidates_idx = attack_utils.get_poisoning_candidate_samples('ember',self.ori_model,self.x_ori,self.y_ori_label)
            self.backdoor_acc, self.ori_test_acc = self.base_test()
        self.selected_feature,self.shap_value_b,self.sel_feat_value = self.data_process()
        self.record = []
        self.louvain_config = config.search_config(self.dataset,self.wm_size,self.poison_rate,self.strategy,self.surrogate_model)

    def build_model(self):
        print(f'[+] Proxy_model not found, Training a proxy model')
        if self.surrogate_model == 'lightgbm' or self.surrogate_model == 'linearsvc':
            ### Non-surrogate explanation not implemented for SVM. This is a temporary solution to use a lightgbm model to compute the shap value of linearsvc model. And it is also used in Severi's work. 
            start = time.perf_counter()
            model = lgb.train({"application": "binary",'verbose':-1}, lgb.Dataset(self.X_train, self.y_train))
            end = time.perf_counter()
            print(f'[-] Building model took {round(end-start,2)} s')
        elif self.surrogate_model == 'RF':
            start = time.perf_counter()
            model = RandomForestClassifier(
                n_estimators=1000,  # Used by PDFrate
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=43,  # Used by PDFrate
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,  # Run in parallel
                random_state=16,
                verbose=0
            ).fit(self.X_train, self.y_train)
            end = time.perf_counter()
            print(f'[-] Building model took {round(end-start,2)} s')
        elif self.surrogate_model == 'DNN':
            start = time.perf_counter()
            model = Deep_NN(135)
            model.fit(self.X_train,self.y_train)
            end = time.perf_counter()
            print(f'[-] Building model took {round(end-start,2)} s')
        return model

    def config_show(self,config_pos):
        config = np.load(config_pos,allow_pickle=True).item()
        feature_values = config['watermark_features'].values()
        feature_ids = config['wm_feat_ids']
        watermark = {feature_ids[indice]:i for indice,i in enumerate(feature_values)}
        sort = sorted(watermark.items(),key=lambda d:d[0])
        return {i[0]:i[1] for i in sort}
    
    def summary(self,version = 1):
        if version == 1:
            print(f'==============================================================================================')
            print(f'==============================================================================================')
        print(f'[*] Target Attack Strategy : {self.strategy}')
        if version == 1:
            print(f'[*] X_train shape : {self.X_train.shape}')
        print(f'[*] watermark size : {self.wm_size}')
        print(f'[*] poison rate : {self.poison_rate}')
        print(f'[*] poison num : {self.poison_num}')
        if self.wm_size != 0:
            print(f'[*] watermark : {self.config}')
            if self.dataset == 'ember' and version == 2:
                print(f'[*] Acc(Fb,Xb) : {self.backdoor_acc}')
        if self.dataset == 'ember' and version == 2:
            print(f'[*] Acc(Fb,Xt) : {self.ori_test_acc}')
        print(f'[*] Surrogate Model : {self.surrogate_model}')
        if version == 1:
            print(f'==============================================================================================')
            print(f'==============================================================================================')

    def search_feature_abs(self,shap_value):
        """
            search important features with shap_value_abs
        """
        importance = {i:0 for i in range(self.length)}
        for i in shap_value:
            for j in range(self.length):
                importance[j] += abs(i[j])
        importance = sorted(importance.items(), key=lambda d: d[1], reverse=True)[:32]
        return importance
    
    def base_test(self):
        """
            shows the  backdoor malware detection acc & ori test set detection acc of backdoor model
        """
        y_pred = self.backdoor_model.predict(self.X_test)
        y_analysis = [1 if i >0.5 else 0 for i in y_pred]
        acc = accuracy_score(y_analysis,[1]*len(y_analysis))
        y_ori_pred = [1 if i >0.5 else 0 for i in self.backdoor_model.predict(self.x_mw_poisoning_candidates)]
        acc1 = accuracy_score([1]*len(y_ori_pred), y_ori_pred)
        return f'{round(acc*100,2)}%', f'{round(acc1*100,2)}%'
    
    def data_process(self):
        """
            calculate shap value of benign samples, and shows the samples with selected features in DataFrame format.
        """
        # test_b = self.X_train[self.middle:-self.poison_num].tolist()
        # test_w = self.X_train[-self.poison_num:].tolist()
        # test = test_b + test_w
        test = self.X_train[self.middle:].tolist()
        # print('[*] Searching feature with shap_abs')
        if self.surrogate_model == 'lightgbm' or self.surrogate_model == 'linearsvc':
            shap_value = self.backdoor_model.predict(test,pred_contrib=True)[:,:-1]
        elif self.surrogate_model == 'RF':
            explainer = shap.TreeExplainer(self.backdoor_model)
            shap_value = explainer.shap_values(np.array(test))[1]
        elif self.surrogate_model == 'DNN':
            shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
            shap_value = self.backdoor_model.explain(X_back = self.X_train , X_exp = test)[0]
        selected_feature = self.search_feature_abs(shap_value)
        sel_feat = [m for (m,n) in selected_feature]
        sel_feat_df = pd.DataFrame(test)
        sel_feat_value = sel_feat_df[sel_feat]
        return [m for (m,n) in selected_feature],shap_value,sel_feat_value
    
    def evaluation(self,X,y):
        """
            Evaluation after retraining a new model without detected poison samples.
            And calculate backdoor malware detection acc and ori test set detection acc to compare with base_line. 
        """
        eva_model = model_utils.train_model(model_id='lightgbm',x_train=X,y_train=y)
        if self.wm_size != 0: 
            y_pred = eva_model.predict(self.X_test)
            y_analysis = [1 if i >0.5 else 0 for i in y_pred]
            acc = accuracy_score(y_analysis,[1]*len(y_analysis))
            print(f'       [*] Acc(Fa,Xb) : {round(acc*100,2)}%')
        y_ori_pred = [1 if i >0.5 else 0 for i in eva_model.predict(self.x_mw_poisoning_candidates)]
        acc1 = accuracy_score([1]*len(y_ori_pred), y_ori_pred)
        print(f'       [*] Acc(Fa,Xt) : {round(acc1*100,2)}%')
        if self.wm_size != 0:    
            return f'{round(acc*100,2)}%',f'{round(acc1*100,2)}%'
        else:
            return None,f'{round(acc1*100,2)}%'
    
    def iso_f(self):
        """
            Isolation Forest : a mutation method proposed in paper
        """
        print('[START] Isolation Forest')
        xtrain = self.sel_feat_value.values
        starttime = time.time()
        isof = IsolationForest(max_samples='auto', contamination='auto', random_state=42, n_jobs=-1)
        isof_pred = isof.fit_predict(xtrain)
        print('       Training the Isolation Forest took {:.2f} seconds'.format(time.time() - starttime))
        
        if self.wm_size != 0:
            TP = isof_pred[-self.poison_num:].tolist().count(-1)
            FN = isof_pred[-self.poison_num:].tolist().count(1)
            FP = isof_pred[:-self.poison_num].tolist().count(-1)
            TN = isof_pred[:-self.poison_num].tolist().count(1)
        else:
            TP = None
            FN = None
            FP = isof_pred.tolist().count(-1)
            TN = isof_pred.tolist().count(1)
        print('       [*] Remove identified watermarked samples:')
        print('       [isolation forest][-] TP : ',TP)
        print('       [isolation forest][+] FN : ',FN)
        print('       [isolation forest][-] FP : ',FP)
        print('       [isolation forest][+] TN : ',TN)

        if self.wm_size != 0:
            TPR = f'{round(TP*100/(TP+FN),2)}%'
        else:
            TPR = None
        FPR = f'{round(FP*100/(FP+TN),2)}%'

        print(f'       [isolation forest] TPR : {TPR}')
        print(f'       [isolation forest] FPR : {FPR}')

        print('       [*] Training a model without detected watermarked samples')

        malware = self.X_train[:self.middle].tolist()
        benign = self.X_train[self.middle:].tolist()
        new_benign = []
        for indice,i in enumerate(isof_pred):
            if i == 1:
                new_benign.append(benign[indice])

        X_isof = np.array(new_benign + malware)
        label_lsof = np.array([0] * len(new_benign) + [1] * len(malware))
        backdoor_acc,ori_acc = self.evaluation(X_isof,label_lsof)
        # backdoor_acc,ori_acc = None,None
        self.record.append([TP,FN,FP,TN,TPR,FPR,backdoor_acc,ori_acc])
        print('[Finish] Isolation Forest\n=======================================================')
    
    def HDBScan(self):
        """
            HDBScan : a mutation method proposed in paper
        """
        print('[START] HDBScan')
        mcs = int(0.005 * self.X_train.shape[0])
        ms = int(0.001 * self.X_train.shape[0])
        x_gw_sel_std = defense_utils.standardize_data(self.sel_feat_value.values.tolist())
        clustering, clustering_labels = defense_utils.cluster_hdbscan(
            data_mat=x_gw_sel_std,
            metric='euclidean',
            min_clus_size=mcs,
            min_samples=ms,
            n_jobs=32,
        )

        is_clean = np.ones(self.middle, dtype=int)
        is_clean[-self.poison_num:] = 0

        silh, avg_silh = defense_utils.compute_silhouettes(data_mat=x_gw_sel_std,labels=clustering_labels)
        cluster_sizes, evals = defense_utils.show_clustering(
            labels=clustering_labels,
            is_clean=is_clean,print_mc=len(set(clustering_labels)),
            print_ev=len(set(clustering_labels)),
            avg_silh=avg_silh)
        
        t_max_size = 0.1 * self.X_train.shape[0]
        min_keep_percentage = 0.2

        expand_silh = np.array([avg_silh[j] if cluster_sizes[j] <= t_max_size else -1 for j in clustering_labels])
        std_silh = defense_utils.standardize_data(data_mat=expand_silh.reshape(-1, 1),feature_range=(0, 1))
        scores = np.ones(std_silh.shape)
        scores = (scores - std_silh) + min_keep_percentage
        np.random.seed(42)
        rand_draw = np.random.random_sample(scores.shape)
        selected = (scores >= rand_draw).flatten()
        print('       Number of removed samples: {}'.format(selected.shape-sum(selected)))
        print('       [*] Remove identified watermarked samples:')
        if self.wm_size != 0:
            TP = selected[-self.poison_num:].tolist().count(False)
            FN = selected[-self.poison_num:].tolist().count(True)
            FP = selected[:-self.poison_num].tolist().count(False)
            TN = selected[:-self.poison_num].tolist().count(True)
        else:
            TP = None
            FN = None
            FP = selected.tolist().count(False)
            TN = selected.tolist().count(True)

        print('       [hdbscan][-] TP : ',TP)
        print('       [hdbscan][+] FN : ',FN)
        print('       [hdbscan][-] FP : ',FP)
        print('       [hdbscan][+] TN : ',TN)
        
        if self.wm_size != 0:
            TPR = f'{round(TP*100/(TP+FN),2)}%'
        else:
            TPR = None
        FPR = f'{round(FP*100/(FP+TN),2)}%'

        print(f'       [hdbscan] TPR : {TPR}')
        print(f'       [hdbscan] FPR : {FPR}')

        print('       [*] Training a model without detected watermarked samples')

        malware = self.X_train[:self.middle].tolist()
        benign = self.X_train[self.middle:].tolist()
        new_benign_hdbscan = []
        for indice,i in enumerate(selected):
            if i == True:
                new_benign_hdbscan.append(benign[indice])

        X_hdbscan = np.array(new_benign_hdbscan + malware)
        label_hdbscan = np.array([0] * len(new_benign_hdbscan) + [1] * len(malware))
        backdoor_acc,ori_acc = self.evaluation(X_hdbscan,label_hdbscan)
        # backdoor_acc,ori_acc = None,None
        self.record.append([TP,FN,FP,TN,TPR,FPR,backdoor_acc,ori_acc])
        print('[Finish] HDBScan\n=======================================================')
    
    def spectral_signatures(self):
        """
            Spectral_Signatures : a mutation method proposed in paper
        """
        print('[START] spectral_signatures')
        x_gw_sel_std = defense_utils.standardize_data(self.sel_feat_value.values.tolist())
        is_clean = np.ones(self.middle, dtype=int)
        is_clean[-self.poison_num:] = 0
        bdr_indices = set(np.argwhere(is_clean == 0).flatten().tolist())
        to_remove_pa, found_pa = defense_utils.spectral_remove_lists(x_gw_sel_std, bdr_indices)

        if self.wm_size != 0:
            TP = to_remove_pa[-self.poison_num:].tolist().count(1)
            FN = to_remove_pa[-self.poison_num:].tolist().count(0)
            FP = to_remove_pa[:-self.poison_num].tolist().count(1)
            TN = to_remove_pa[:-self.poison_num].tolist().count(0)
        else:
            TP = None
            FN = None
            FP = to_remove_pa.tolist().count(1)
            TN = to_remove_pa.tolist().count(0)
        print('       [*] Remove identified watermarked samples:')
        print('       [spectral signature][-] TP : ',TP)
        print('       [spectral signature][+] FN : ',FN)
        print('       [spectral signature][-] FP : ',FP)
        print('       [spectral signature][+] TN : ',TN)

        if self.wm_size != 0: 
            TPR = f'{round(TP*100/(TP+FN),2)}%'
        else:
            TPR = None
        FPR = f'{round(FP*100/(FP+TN),2)}%'

        print(f'       [spectral signature] TPR : {TPR}')
        print(f'       [spectral signature] FPR : {FPR}')
        print('       [*] Training a model without detected watermarked samples')
        malware = self.X_train[:self.middle].tolist()
        benign = self.X_train[self.middle:].tolist()
        new_benign_ss = []
        for indice,i in enumerate(to_remove_pa):
            if i == 0:
                new_benign_ss.append(benign[indice])

        X_ss = np.array(new_benign_ss + malware)
        label_ss = np.array([0] * len(new_benign_ss) + [1] * len(malware))
        backdoor_acc,ori_acc = self.evaluation(X_ss,label_ss)
        # backdoor_acc,ori_acc = None,None
        self.record.append([TP,FN,FP,TN,TPR,FPR,backdoor_acc,ori_acc])
        print('[Finish] spectral_signatures\n=======================================================')
    
    def MDR(self):
        """
            Our Mutation Method : Explanation Powered Defense
        """
        print('[START] MDR')
        print('       [MDR] Start Suspicious Samples Filtering')
        print(f'       [+] Creating features with the strongly goodware-oriented features')
        time.sleep(.5)
        filter = VarianceThreshold(threshold=0.5)
        filter.fit_transform(self.X_train)
        filter_feature = filter.get_support(indices=True)

        feature = self.create_feature(self.X_train[self.middle:],self.shap_value_b,filter_feature)

        print(f'       [+] Building Graph based on Intersection Similarity')
        G = self.build_Graph(feature)

        print(f'       [+] Starting Community Division')
        partition = community_louvain.best_partition(G,random_state=1)
  
        print(f'       [+] Searching the target community using anti-pertubation elements')
        cluster_w = self.choose_cluster(partition,feature)

        cluster = [i for i in partition if partition[i] == cluster_w]
        cls = [0] * self.middle
        for i in cluster:
            cls[i] = 1

        if self.wm_size != 0:
            TP = cls[-self.poison_num:].count(1)
            FN = self.poison_num - cls[-self.poison_num:].count(1)
            FP = cls[:-self.poison_num].count(1)
            TN = self.middle- self.poison_num -cls[:-self.poison_num].count(1)
        else:
            TP = None
            FN = None
            FP = cls.count(1)
            TN = self.middle- self.poison_num -cls.count(1)

        print('       [Suspicious Samples Filtering][-] TP : ',TP)
        print('       [Suspicious Samples Filtering][+] FN : ',FN)
        print('       [Suspicious Samples Filtering][-] FP : ',FP)
        print('       [Suspicious Samples Filtering][+] TN : ',TN)
        
        if self.wm_size != 0:
            TPR = f'{round(TP*100/(TP+FN),2)}%'
        else:
            TPR = None
        FPR = f'{round(FP*100/(FP+TN),2)}%'

        print(f'       [Suspicious Samples Filtering] TPR : {TPR}')
        print(f'       [Suspicious Samples Filtering] FPR : {FPR}')

        # self.record.append([TP,FN,FP,TN,TPR,FPR,None,None])
        print('       [Finish] Suspicious Samples Filtering\n       =======================================================')

        """
        Watermark Identify : Further step of Data Filtering
        """
        print('       [MDR] Start Watermark Identification')
        scanner = wm_identify(cluster,self.backdoor_model,self.X_train[self.middle:],filter_feature,self.shap_value_b,self.middle, self.length)
        print(f'       [+] Scanning Watermark')
        scanner.search_wm(cluster)
        watermark = scanner.feature
        watermark_display = {i:watermark[i][1] for i in watermark}
        if self.wm_size != 0:
            num_wm_correct = len(set([(i,watermark_display[i]) for i in watermark_display]) & set([(j,self.config[j]) for j in self.config]))
            self.wm_i = len(watermark_display)
            self.wm_i_c = num_wm_correct
            print(f'       [*] Found suspicious watermark :\n{watermark_display}')
            print(f'       [+] Number of identified watermark features: {len(watermark_display)}')
            print(f'       [+] Number of correct identified watermark features : {num_wm_correct}')
            print(f'       [Watermark Identification] Pwm : {round(num_wm_correct*100/len(watermark_display),2)}%')

        print('       [Finish] Watermark Identification\n       =======================================================')

        remove_cls = self.remove_after_identity(watermark)

        if self.wm_size != 0:
            MDR_TP = remove_cls[-self.poison_num:].tolist().count(1)
            MDR_FN = self.poison_num - remove_cls[-self.poison_num:].tolist().count(1)
            MDR_FP = remove_cls[:-self.poison_num].tolist().count(1)
            MDR_TN = self.middle- self.poison_num -remove_cls[:-self.poison_num].tolist().count(1)
        else:
            MDR_TP = None
            MDR_FN = None
            MDR_FP = remove_cls.tolist().count(1)
            MDR_TN = self.middle- self.poison_num -remove_cls.tolist().count(1)

        print('       [MDR][-] TP : ',MDR_TP)
        print('       [MDR][+] FN : ',MDR_FN)
        print('       [MDR][-] FP : ',MDR_FP)
        print('       [MDR][+] TN : ',MDR_TN)

        if self.wm_size != 0:
            MDR_TPR = f'{round(MDR_TP*100/(MDR_TP+MDR_FN),2)}%'
        else:
            MDR_TPR = None
        MDR_FPR = f'{round(MDR_FP*100/(MDR_FP+MDR_TN),2)}%'
        
        print(f'       [Suspicious Samples Filtering] TPR : {MDR_TPR}')
        print(f'       [Suspicious Samples Filtering] FPR : {MDR_FPR}')
        new_mal = self.create_mal(watermark_display)
        if self.wm_size == 0:
            y_new_mal_pred = [1 if i >0.5 else 0 for i in self.backdoor_model.predict(new_mal)]
            asr = accuracy_score([0]*len(y_new_mal_pred), y_new_mal_pred)
            print(f'       Attack success rate : {round(asr*100,2)}%')
        
        print('       [*] Training a model using samples without identified watermark features')
        new_x_pro = self.X_train[self.middle:][remove_cls!=1].tolist()
        new_train_pro = np.array(new_x_pro + self.X_train[:self.middle].tolist() + new_mal.tolist())
        new_label_pro = np.array([0]*len(new_x_pro) + [1]*len(self.X_train[:self.middle]) + [1]*len(new_mal))

        if self.dataset == 'ember':
            backdoor_acc_pro,ori_acc_pro = self.evaluation(new_train_pro,new_label_pro)
        # backdoor_acc_pro,ori_acc_pro = None,None
            self.record.append([MDR_TP,MDR_FN,MDR_FP,MDR_TN,MDR_TPR,MDR_FPR,backdoor_acc_pro,ori_acc_pro])
        else:
            self.record.append([MDR_TP,MDR_FN,MDR_FP,MDR_TN,MDR_TPR,MDR_FPR])
            dirname = os.path.dirname(self.X_train_pos)
            saved_X_path = os.path.join(dirname,f'MDR_X_{self.surrogate_model}.npy')
            saved_y_path = os.path.join(dirname,f'MDR_y_{self.surrogate_model}.npy')
            if not os.path.exists(saved_X_path):
                np.save(saved_X_path,new_train_pro)
                np.save(saved_y_path,new_label_pro)
        print('[Finish] MDR')
    
    def create_mal(self,wm):
        tmp = []
        mal = self.X_train[:100].copy()
        for i in mal.copy():
            for j in wm:
                i[j] = wm[j]
            tmp.append(i)
        return np.array(tmp)
    
    def build_Graph(self,feature):
        # print(f"       [+] Using threshold : {self.louvain_config['threshold']}")
        time.sleep(.5)
        G = nx.Graph()
        for idx1,i in enumerate(feature):
            for idx2 in range(idx1+1,len(feature)):
                dis = len(set(feature[idx1])&set(feature[idx2]))
                if dis >=self.louvain_config['threshold']:
                    G.add_edge(idx1,idx2,weight=51-dis)
        return G


    def remove_after_identity(self,watermark):
        tmp = []
        print('       [*] Remove identified watermarked samples:')
        for m in self.X_train[self.middle:]:
            if [watermark[k][1] for k in watermark] == [m[j] for j in watermark]:
                tmp.append(1)
            else:
                tmp.append(0)
        return np.array(tmp)

    
    def choose_cluster(self,partition,feature):
        """
            @params:
                data : input of KMeans
                cls : label of classification
            @return:
                the cluster id which may contain watermark samples
        """
        cls = [0] * self.middle
        for i in partition:
            cls[i] = partition[i]
        _tmp = {}
        for i in cls:
            if i in _tmp:
                _tmp[i] += 1
            else:
                _tmp[i] = 1
        result = sorted(_tmp.items(),key=lambda d:d[1],reverse=True)
        if self.wm_size != 0:
            clusters = [i[0] for i in result if (i[0]!=0 and i[1] > self.poison_num-50 and i[1] < 500)]
        else:
            clusters = [i[0] for i in result if (i[0]!=0 and i[1] > 50 and i[1] < 500)]
        
        tmp = {}
        for cluster_id in clusters:
            cluster_id_feature = np.array(feature,dtype=object)[np.array([indice for indice,i in enumerate(cls) if i == cluster_id])]
            tmp[cluster_id] = self.apply_pertub(cluster_id_feature)
        return sorted(tmp.items(),key=lambda d:d[1])[0][0]


    def apply_pertub(self,cluster_id_feature):
        """
        @para :
            x : [n1,n2,n3,n3] .x represents the indice of samples need to be apply pertub
            mal : malware sample applied to x
            shap_value : shap value of whole benign dataset
            num : number of feature to be applied to pertub
        """
        tmp = {}
        for k in cluster_id_feature:
            # _k = [(m,k[m]) for m in k]
            for y in k:
                if y in tmp:
                    tmp[y] += 1
                else:
                    tmp[y] = 1
        tmp = [(m[0],m[1]) for m in tmp if tmp[m] > len(cluster_id_feature)*0.9]
        pertub_ele = {i[0]:i[1] for i in tmp}
        need_pertub = self.X_train[:100].copy()
        for j in pertub_ele:
            need_pertub[:,j] = pertub_ele[j]
        return np.median(self.backdoor_model.predict(need_pertub))
    
    def create_feature(self,x,shap_x,fea):
        """
            @params:
                x : benign samples from X_train
                shap_x : shap_value of x
                num = k : the number of the first k important features
            @return:
                the first k important features for each sample in x
        """
        feature = []
        for indice,i in enumerate(x):
            shap_value = shap_x[indice]
            importance = {indice2:j for indice2,j in enumerate(shap_value) if j<- 0.01}
            feature_sel = [(m,i[m]) for m,n in sorted(importance.items(), key=lambda d: d[1]) if m in fea]
            feature.append(feature_sel)
        return feature

    def run_defense(self):
        """
            run three mutation methods mentioned in paper, and one mutation from our work:EPD(Explaination Powered Defense)
        """
        self.iso_f()
        self.HDBScan()
        self.spectral_signatures()
        self.MDR()
    
    def show(self):
        print(f'----------------Overview : Defense against the attack with {self.poison_rate*100}%_{self.strategy}----------------')
        self.summary(version=2)
        if self.dataset == 'ember':
            col = ['TP','FN','FP','TN','TPR','FPR','Acc(Fa,Xb)','Acc(Fa,Xt)']
            vis = pd.DataFrame(self.record,columns=col,index=['Isolation Forest','HDBScan','Spectral Signature','MDR'])
        else:
            col = ['TP','FN','FP','TN','TPR','FPR']
            vis = pd.DataFrame(self.record,columns=col,index=['MDR'])
        return vis

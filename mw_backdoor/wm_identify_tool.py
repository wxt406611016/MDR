"""
@author: supervampire
@email: wangxutong@iie.ac.cn
@version: Created in 2022 03/25.
"""
import itertools
import numpy as np

class wm_identify:
    def __init__(self,cluster, backdoor_model, X, filter_feature, shap_value_b, middle, length):
        self.cluster = cluster
        self.backdoor_model = backdoor_model
        self.X = X
        self.others = self.X[np.array(list(set(list(range(middle))) - set(self.cluster)))]
        self.feature = {}
        self.saved = []
        self.filter_feature = filter_feature
        self.shap_value = shap_value_b
        self.middle = middle
        self.length = length
        self.score = 0 
        shap_value_sel = {i:np.mean(self.shap_value[self.cluster][:,i]) for i in range(self.length)}
        self.shap_value_sel = sorted(shap_value_sel.items(),key=lambda d:d[1])
        # print(self.shap_value_sel)

    def count_hit(self,i,heatmap,x):
        """
        @params:
            i: an instance of watermark candidate
            heatmap: watermark feature and value candidate
            x: outer data
        @return:
            the number of samples that matching wm candidate
        """
        count = 0
        for m in x:
            if [heatmap[k][1] for k in i] == [m[j] for j in i] and [self.feature[t][1] for t in self.feature] == [m[q] for q in self.feature]:
                count += 1
        return count
                

    def tf_idf(self,heatmap,others,us,generator):
        """
        @params:
            heatmap:watermark feature and value candidate
            others: number of samples matching wm candidate in outer data
            us: number of samples matching wm candidate in inner data
            generator: a generator that generates instances of watermark candidate
        @return:
            tf-idf
        """
        tmp = {}
        for i in generator:
            a = self.count_hit(i,heatmap,us)
            b = self.count_hit(i,heatmap,others)
            if b == 0:
                b = 0.5
            ti = a/b
            tmp[ti] = i
        tmp_score = sorted(tmp.items(),key = lambda d:d[0],reverse=True)[0][0]
        if tmp_score >= self.score: 
            self.score = tmp_score     
            return tmp
        else:
            return {}

    def count_wm_feature(self,x,wm_feature):
        """
        @params:
            x: data.T
            wm_feature: wm_feature candidate ([a,b,c,d])
        @return:
            watermark
        """
        heatmap = {}
        for i in wm_feature:
            tmp = x[i]
            _tmp = {}
            for j in tmp:
                if j in _tmp:
                    _tmp[j] += 1
                else:
                    _tmp[j] = 1
            _tmp = sorted(_tmp.items(), key=lambda d: d[1],reverse=True)[:1]
            heatmap[i] = [_tmp[0][0],_tmp[0][1]]
        heatmap = sorted(heatmap.items(), key=lambda d: d[1][1],reverse=True)
        heatmap = {i[0]:(i[1][1],i[1][0]) for i in heatmap}
        return heatmap

    def count_wm(self,wm,x):
        """
        @parmas:
            wm: watermark candidate
            x: whole data
        @return:
            list: indice of selected samples
        """
        tmp = []
        for m in x:
            if [wm[k][1] for k in wm] == [self.X[m][j] for j in wm]:
                tmp.append(m)
        return tmp

    def generator(self,heatmap):
        generator = []
        for num in range(int(len(heatmap)/2),len(heatmap)+1):
            generator += [i for i in itertools.combinations([j for j in heatmap], num)]
        return generator 

    def search_wm(self,x,feature_used=[],count=1,k=0):
        """
        @params :
            x : the indice of data that need to be checked,with watermark feature and value (list[1,0,0,1,,,0,0,1])
        @return :
            watermark
        """
        if count % 30 == 1:
            print(f'       [+] Recursive for the {count} round')
        ours = self.X[x]
        shap_value_sel = self.shap_value_sel

        ####
        if count == 1:
            wm_feature = list((set([m for m,n in shap_value_sel[:count*8]]) - set(feature_used)))
        elif count*8 > len(shap_value_sel):
            wm_feature = list((set([m for m,n in shap_value_sel[(count-1)*8:]]) - set(feature_used)))
        else:
            wm_feature = list((set([m for m,n in shap_value_sel[(count-1)*8:count*8]]) - set(feature_used)))
        ####

        heatmap = self.count_wm_feature(ours.T,wm_feature)
        _wm_length = ours.shape[0] * (0.8 + k * 0.1)
        wm_length = _wm_length if k<2 else ours.shape[0]
      
        heatmap = {i:heatmap[i] for i in heatmap if (heatmap[i][0] >= wm_length) and heatmap[i][1]!=0}
        
        _tf_idf = self.tf_idf(heatmap,self.others,ours,self.generator(heatmap))
        _tf_idf_sorted = sorted(_tf_idf.items(),key=lambda d:d[0],reverse=True)
        # print(_tf_idf_sorted)
        try:
            wm = list({i[0]:i[1] for i in _tf_idf_sorted}.values())[0]
            wm = {i:heatmap[i] for i in wm}
        except Exception as e:
            wm = {}
        self.feature.update(wm)
        feature_used = [m for m in self.feature]
        new_x = self.count_wm(wm,x)
        self.saved = new_x
        if count*8 < len(shap_value_sel):
            count += 1
            k += 1
            self.search_wm(new_x,feature_used,count,k)
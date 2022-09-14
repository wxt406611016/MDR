def search_config(dataset,wm_size,p_r,tactic,model):
    if dataset == 'ember':
        if wm_size == 0:
            return {'threshold':10}
        elif wm_size == 8:
            if p_r == 0.01:
                if tactic == 'combined':
                    return {'threshold':9}
                elif tactic == 'independent':
                    return {'threshold':7}
            elif p_r == 0.02:
                if tactic == 'combined':
                    return {'threshold':9}
                elif tactic == 'independent':
                    return {'threshold':6}
            elif p_r == 0.04:
                if tactic == 'combined':
                    return {'threshold':9}
                elif tactic == 'independent':
                    return {'threshold':7}
        elif wm_size == 17:
            if p_r == 0.01:
                if tactic == 'combined':
                    return {'threshold':9}
                elif tactic == 'independent':
                    return {'threshold':8}
            elif p_r == 0.02:
                if tactic == 'combined':
                    return {'threshold':11}
                elif tactic == 'independent':
                    return {'threshold':8}
            elif p_r == 0.04:
                if tactic == 'combined':
                    return {'threshold':10}
                elif tactic == 'independent':
                    return {'threshold':8}
    elif dataset == 'contagio':
        if model == 'lightgbm' or model == 'linearsvc':
            if p_r == 0.02:
                return {'threshold':15}
            elif p_r == 0.08:
                return {'threshold':14}
        elif model == 'RF':
            if p_r == 0.02:
                return {'threshold':7}
            elif p_r == 0.08:
                return {'threshold':7}
        elif model == 'DNN':
            if p_r == 0.02:
                return {'threshold':10}
            elif p_r == 0.08:
                return {'threshold':7}
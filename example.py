import pandas as pd
import numpy as np
from math import pow
from lyhyper import lyHyper
pd.set_option('display.width',2000)
from hyperopt import fmin,hp

def load_data():

    from sklearn.datasets import load_iris

    iris = load_iris()
    feature = iris.data
    label = iris.target
    feature_name = iris.feature_names
    # print(feature)
    data = pd.DataFrame(feature)
    data.columns = feature_name
    data['target'] = label
    return data


if __name__ == '__main__':
    data = load_data()
    data= data.sample(150)
    from sklearn.ensemble import RandomForestClassifier

    # 模型包装函数f(parm,X,y)
    # 需return对应的评价指标
    def rf(parm,X,y):
        rf = RandomForestClassifier(n_estimators=parm['n_estimators'],
                                   oob_score=parm['oob_score'],
                                   max_features=parm['max_features'],
                                   max_depth=parm['max_depth'],
                                   min_samples_split=parm['min_samples_split'],
                                   min_samples_leaf=parm['min_samples_leaf'],
                                   n_jobs=5)

        rf.fit(X[0:100],y[0:100])
        scores = rf.score(X[100:150],y[100:150])
        return scores


    def _parm_base():
        seed = {'n_estimators': 3,
                     'oob_score': False,
                     'max_features': 1,
                     'max_depth': 7,
                     'min_samples_split': 2,
                     'min_samples_leaf': 1}

        parmRange = {'n_estimators':range(1,10,1)}

        hyper= lyHyper(seed=seed,space=parmRange)
        hyper.fit(data.ix[:,0:4],data.ix[:,4],fn=rf,ktype='base',show=True,assess='max',silent=False)
        print('best_pram')
        print(hyper.bestparm)


    def _parm_opt():
        seed = {'n_estimators': 3,
                'oob_score': False,
                'max_features': 1,
                'max_depth': 7,
                'min_samples_split': 2,
                'min_samples_leaf': 1}

        parmRange = {'n_estimators': hp.choice('n_estimators',range(1,10,1))}

        hyper = lyHyper(seed=seed, space=parmRange)
        hyper.fit(data.ix[:, 0:4], data.ix[:, 4], fn=rf, ktype='opt', show=True, assess='max',silent=True)
        print('best_pram')
        print(hyper.bestparm)

    _parm_base()
# coding:utf-8
import numpy as np
import pandas as pd
import os
from hyperopt import fmin,tpe

###########################
# author: allenYang       #
# update: 2017-08-14      #
# email: allenyzx@163.com #
###########################
"""
BASE思路：
:param:模型的parameter
:return:模型的评估值
基于param迭代参数，每一轮只轮询一个参数，在轮到最后参数后取最大值的return的参数，固定该参数，进行其他迭代，类推。

TPE思路：


"""



class lyHyper:

    def __init__(self,space,seed):
        # 需要的参数选择范围
        self.space = space
        # 种子范围
        self.seed = seed
        self.pdoutput = pd.DataFrame()
        self.silentNum = 1
        self.silentStr = '[Count]the %s(%s) is running: %s'

    def _base(self,X,y,fn,assess,silent,show):
        for key in self.space:
            assessDic = {}
            last = self.space[key][-1]

            for changed in self.space[key]:
                self.seed[key] = changed
                modelAssess = fn(X=X,y=y,parm=self.seed)

                output = self.seed
                output['assess'] = modelAssess

                # 显示次数
                if silent is False:
                    self.pdoutput = self._store(self.pdoutput,output)
                    print(self.silentStr%('base',assess,self.silentNum))
                    self.silentNum += 1

                assessDic[assess] = changed
                if changed == last:
                    if assess == 'min':
                        assessValue = assessDic[min(assessDic.keys())]
                    elif assess == 'max':
                        assessValue = assessDic[max(assessDic.keys())]
                    else:
                        raise ValueError('assess parm error')

            self.seed[key] = assessValue

        # 展示结果
        if show is True:
            print(self.pdoutput)

        return


    def _opt(self,X,y,fn,algo,max_evals,trials,assess,silent,show):
        model = fn
        seed = self.seed

        def _obj(parm,fn=model):

            for key in parm.keys():
                seed[key] = parm[key]
            scores = fn(seed, X, y)

            # 存储并显示次数
            store = seed
            store['assess'] =scores
            self.pdoutput = self._store(self.pdoutput,store)
            if silent is False:
                print(self.silentStr % ('opt', assess, self.silentNum))
                self.silentNum += 1

            # 如果评价方向是max，那么scores取反
            scores = - scores if assess == 'max' else scores
            return scores

        if trials == '':
            self._the_best = fmin(_obj, self.space, algo=algo, max_evals=max_evals)
        else:
            self._the_best = fmin(_obj,self.space,algo=algo,max_evals=max_evals,trials=trials)

        # 展示
        if show == True:
            print(self.pdoutput)

        return self._the_best


    def fit(self,X,y,fn,assess,ktype='base',algo=tpe.suggest,max_evals=100,trials='',save=False,csv_path='',silent=False,show=False):
        """
        :param X: 特征矩阵 
        :param y: 标签
        :param fn: 包装函数fn(parm,X,y)，将模型包装成函数，需return评价指标结果(若有多参数除parm,X,y需要默认参数输入)
        :param assess: 可选（max，min），评价函数的方向
        :param ktype: 可选（base，opt），base为坐标下降法，opt为hyperopt
        :param algo: 在opt下的可优化算法，默认tpe.suggest
        :param max_evals: 在opt下的最大迭代次数，默认100
        :param trials: 在opt下数据集抽样
        :param save: 是否存储（True，False）
        :param csv_path: 存储路径（默认tmp.csv）
        :param silent: 是否输出迭代次数，默认False
        :param show: 是否展现结果
        :return: 
        """

        # 存储
        self.assess = assess
        self.ktype = ktype

        # 选择类型
        if ktype == 'base':
            self._base(X=X,y=y,fn=fn,assess=assess,show=show,silent=silent)
        elif ktype == 'opt':
            self._opt(X=X,y=y,fn=fn,show=show,algo=algo,max_evals=max_evals,trials=trials,assess=assess,silent=silent)
        else:
            raise ValueError('ktype parm error')

        # 存储评价数据
        if save == True:
            if csv_path != '':
                self._save_csv(pdoutput=self.pdoutput, csv_path=csv_path)
            else:
                local_path = os.path.dirname(__file__)
                self._save_csv(pdoutput=self.pdoutput,csv_path=local_path+'/tmp.csv')
        return

    @property
    def bestparm(self):

        if self.assess == 'min':
            _filter = self.pdoutput['assess'].min()
        elif self.assess == 'max':
            _filter = self.pdoutput['assess'].max()
        else:
            raise ValueError('assess set error')

        df = self.pdoutput[self.pdoutput['assess']==_filter]

        # 由于hyperopt由算法得到参数可能跟原始轮询结果出来的不一样，有已知轮询(known)和hyperopt(hyper)两种结果
        if self.ktype == 'opt':
            opt_pram = {}
            opt_pram['hyper'] = self._the_best
            opt_pram['known'] = df.to_dict(orient='index')
            return opt_pram
        elif self.ktype == 'base':
            return df

        else:
            raise ValueError('ktype parm error')

    def _save_csv(self,pdoutput,csv_path):
        return pdoutput.to_csv(csv_path,index=False,encoding='GBK')

    def _store(self,pdoutput,dic):
        df = pd.DataFrame().from_dict(dic,orient='index').T
        pdoutput = pdoutput.append(df)
        return pdoutput.reset_index(drop=True)

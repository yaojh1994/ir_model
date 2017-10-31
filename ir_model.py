#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from datetime import datetime
from calendar import monthrange
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.style.use('seaborn')

class Ir_pre(object):
    def __init__(self):
        self.ir = self.preprocessing()
        self.indic = self.cal_indic()

    def preprocessing(self, plot = False):
        ir = pd.read_csv('ir.csv', index_col = 0, parse_dates = True)
        ir = ir.replace(0.0, np.nan)
        ir = ir.fillna(method = 'pad')
        ir = ir.fillna(0.0)
        ir = ir.groupby(ir.index.strftime('%Y-%m')).last()
        ir.index = [datetime(int(x[:4]),int(x[-2:]),monthrange(int(x[:4]),int(x[-2:]))[1],) for x in ir.index]
        ir = ir.loc['2005-12':'2017-09']
        pl = self.pl_interp()
        ir['pl'] = pl.pl
        ir = ir.dropna()

        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ir1 = 'bond'
            ir2 = 'pl'
            date_range = slice('2005-12','2017-10')
            ax1.plot(ir.loc[date_range, [ir1]], color = 'b', label = ir1, alpha = 0.5, linewidth = 2)
            ax2.plot(ir.loc[date_range, [ir2]], color = 'y', label = ir2, alpha = 0.5, linewidth = 2)
            ax1.legend(loc = 2)
            ax2.legend(loc = 1)
            plt.show()
        return ir

    def cal_indic(self):
        nci = pd.read_csv('nci.csv', index_col = 0, parse_dates = True)
        nci = nci.groupby(nci.index.strftime('%Y-%m')).sum()
        nci.index = [datetime(int(x[:4]),int(x[-2:]),monthrange(int(x[:4]),int(x[-2:]))[1],) for x in nci.index]
        nci = nci.rolling(12).sum()
        #nci.to_csv('nci.csv',sep = ',', index_label = 'date')

        rr = pd.read_csv('rr.csv', index_col = 0, parse_dates = True)

        eri = pd.read_csv('eri.csv', index_col = 0, parse_dates = True)
        eri = eri.resample('m').last()
        eri = 1/eri

        m1 = pd.read_csv('m1.csv', index_col = 0, parse_dates = True)
        m2 = pd.read_csv('m2.csv', index_col = 0, parse_dates = True)
        gdp = pd.read_csv('gdp.csv', index_col = 0, parse_dates = True)
        gdp = gdp.resample('m').fillna(method = 'pad')
        m1_gdp = pd.read_csv('m1_gdp.csv', index_col = 0, parse_dates = True)
        m1_gdp = m1_gdp.resample('m').fillna(method = 'pad')
        m1_gdp = m1_gdp.pct_change(12)
        cpi = pd.read_csv('cpi.csv', index_col = 0, parse_dates = True)
        ppi = pd.read_csv('ppi.csv', index_col = 0, parse_dates = True)
        sf = pd.read_csv('sf.csv', index_col = 0, parse_dates = True)
        sf = sf.pct_change(12)
        cgpi = pd.read_csv('cgpi.csv', index_col = 0, parse_dates = True)
        cgpi = cgpi - 100
        dr7 = pd.read_csv('dr7.csv', index_col = 0, parse_dates = True)
        dr7 = dr7.resample('m').last()
        tdr1 = pd.read_csv('tdr1.csv', index_col = 0, parse_dates = True)
        tdr1 = tdr1.resample('m').fillna(method = 'pad')
        ltlp = pd.read_csv('ltlp.csv', index_col = 0, parse_dates = True)
        ltlp = ltlp.pct_change(12)
        pmi = pd.read_csv('pmi.csv', index_col = 0, parse_dates = True)
        ib7 = pd.read_csv('ib7.csv', index_col = 0, parse_dates = True)
        ib7 = ib7.resample('m').last()
        ib30 = pd.read_csv('ib30.csv', index_col = 0, parse_dates = True)
        ib1y = pd.read_csv('ib1y.csv', index_col = 0, parse_dates = True)
        ib1y = ib1y.resample('m').last()
        fai_source = pd.read_csv('fai_source.csv', index_col = 0, parse_dates = True)
        fai_source = fai_source.pct_change(11).resample('m').fillna(method='pad')
        fai_complete = pd.read_csv('fai_complete.csv', index_col = 0, parse_dates = True)
        fai_complete = fai_complete.pct_change(11).resample('m').fillna(method='pad')
        fai = pd.DataFrame(fai_source.fai_source-fai_complete.fai_complete,columns = ['fai'])
        fai = fai.rolling(12).mean()
        ie = pd.read_csv('ie.csv', index_col = 0, parse_dates = True)
        ie = ie.resample('m').fillna(method = 'pad')
        fe = pd.read_csv('fe.csv', index_col = 0, parse_dates = True)
        fe = fe.resample('m').fillna(method = 'pad')
        ubond = pd.read_csv('ubond.csv', index_col = 0, parse_dates = True)
        ubond = ubond.resample('m').last()
        sz = pd.read_csv('sz.csv', index_col = 0, parse_dates = True)
        #fai = fai.shift(14)

        #ltlp = ltlp.resample('m').fillna(method = 'pad')
        #ltlp = ltlp.pct_change(12)

        indic = pd.concat([nci,rr,eri,m1,m2,gdp,m1_gdp,cpi,ppi,cgpi,dr7,tdr1,\
                ltlp,pmi,ib7,ib30,ib1y,fai_source,fai_complete,fai,ie,fe,ubond,\
                sz], \
                1)['2005-12':'2017-09'].fillna(method = 'pad')
        indic['bond'] = self.ir['bond']
        indic['pl'] = self.ir['pl']
        indic['bond_ubond'] = indic.bond - indic.ubond
        indic['bond_cpi'] = indic.bond - indic.cpi
        indic['m2_gdp'] = indic.m2/indic.gdp
        #indic['ltlp_m2'] = indic['ltlp']*indic['m2']
        self.gdp = gdp
        self.m1 = m1
        self.m1_gdp = m1_gdp
        self.m2 = m2
        self.fai = fai
        self.fai_source = fai_source
        self.fai_complete = fai_complete
        return indic

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ir1 = 'bond'
        ir2 = 'm2_gdp'

        indic1 = self.indic.loc[:, [ir1]]
        indic2 = -self.indic.loc[:, [ir2]]

        ax1.plot(indic1, color = 'b', label = ir1, alpha = 0.5, linewidth = 2)
        ax2.plot(indic2, color = 'y', label = ir2, alpha = 0.5, linewidth = 2)
        ax1.legend(loc = 2)
        ax2.legend(loc = 1)
        plt.show()
        print np.corrcoef(indic1.values.ravel(), indic2.values.ravel())

    def training(self):
        train_slice = slice('2005','2014')
        dates = pd.date_range('2017-10-31', '2019-10-31', freq = 'm')
        gdp = self.gdp
        m2 = self.m2
        m1 = self.m1
        m1_gdp = self.m1_gdp
        fai = self.fai
        fai_source = self.fai_source
        fai_complete = self.fai_complete
        y = self.indic.loc[:, ['bond']]
        for date in dates:
            gdp.loc[date] = np.nan
            m2.loc[date] = np.nan
            m1.loc[date] = np.nan
            m1_gdp.loc[date] = np.nan
            fai_source.loc[date] = np.nan
            fai_complete.loc[date] = np.nan
            fai.loc[date] = np.nan
            y.loc[date] = np.nan
        y = y.fillna(method = 'pad')
        gdp = gdp.shift(24).loc['2005-12':]
        m2 = m2.shift(24).loc['2005-12':]
        m1 = m1.shift(24).loc['2005-12':]
        m1_gdp = m1_gdp.shift(24).loc['2005-12':]
        fai = fai.shift(6).loc['2005-12':]
        fai_source = fai_source.shift(12).loc['2005-12':]
        fai_complete = fai_complete.shift(12).loc['2005-12':]

        x = pd.concat([fai], 1)
        x = x.dropna()
        x_train = x.loc[train_slice]
        y_train = y.loc[train_slice]
        #lr = LogisticRegression()
        lr = LinearRegression()
        model = lr.fit(x_train, y_train)
        print 'model.coef:', model.coef_
        print 'model.socre:', model.score(x_train, y_train)

        #result = model.predict_proba(x)[:,1]
        result = model.predict(x)
        result_df = pd.DataFrame(result, columns = ['fit_value'], index = x.index)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        #ax1.plot(self.x.loc[:, ['gap']], color = 'b')
        #ax2.bar(result_df[:'2014'].index, result_df[:'2014'].values.ravel(), width = 20, alpha = 0.5, color = 'y')
        #ax2.bar(result_df['2015':].index, result_df['2015':].values.ravel(), width = 20, alpha = 0.5, color = 'r')
        ax1.plot(y, color = 'b')
        ax2.plot(result_df[:'2014'].index, result_df[:'2014'].values.ravel(), color = 'y', linewidth = 2)
        ax2.plot(result_df['2014-11-30':].index, result_df['2014-11-30':].values.ravel(), color = 'r', linewidth = 2)
        plt.axvline('2017-09-30', color = 'k', linewidth = 1)

        plt.show()

    def cal_wacc(self):
        rb = pd.read_csv('rb.csv', index_col = 0, parse_dates = True)
        rs = pd.read_csv('rs.csv', index_col = 0, parse_dates = True)
        lr = pd.read_csv('lr.csv', index_col = 0, parse_dates = True)
        sz = pd.read_csv('sz.csv', index_col = 0, parse_dates = True)

        for year in range(2003, 2018):
            if year < 2017:
                for month in range(1, 13):
                    rb['%d-%d'%(year, month)]+= rb['%d-%d'%(year-1, 12)].values[0][0]
            else:
                for month in range(1, 10):
                    rb['%d-%d'%(year, month)]+= rb['%d-%d'%(year-1, 12)].values[0][0]

        for year in range(2003, 2018):
            if year < 2017:
                for month in range(1, 13):
                    rs['%d-%d'%(year, month)]+= rs['%d-%d'%(year-1, 12)].values[0][0]
            else:
                for month in range(1, 10):
                    rs['%d-%d'%(year, month)]+= rs['%d-%d'%(year-1, 12)].values[0][0]

        lr = lr/100
        wb = pd.DataFrame(rb.rb/(rb.rb+rs.rs), columns = ['wb'])
        s = sz.pct_change(12)
        df = pd.concat([s, lr, wb],1)
        df = df.dropna()

        df['wacc'] = df['sz']*(1-df['wb']) +  df['lr']*df['wb']*(1-0.25)
        print df
        #df[['wacc']].to_csv('wacc.csv', index_label = 'date')
        df[['wacc']].plot()
        plt.show()

    def pl_interp(self):
        pl = pd.read_csv('pl.csv', index_col = 0, parse_dates = True)
        #pl.index = [datetime(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[2])) for x in pl.index]
        pl = pl.resample('m').last()
        plv = pl.values.ravel()
        x_all = np.arange(len(plv))
        x_interp = x_all[~np.isnan(plv)]
        y_interp = plv[~np.isnan(plv)]
        f = interp1d(x_interp, y_interp, kind = 'cubic')
        y = f(x_all)
        pl['pl'] = y
        '''
        pl.plot()
        plt.show()
        '''
        return pl

if __name__ == '__main__':
    ir_pre = Ir_pre()
    ir_pre.preprocessing()
    ir_pre.plot()
    #ir_pre.training()

    #ir_pre.cal_indic()
    #ir_pre.cal_wacc()
    #ir_pre.pl_interp()

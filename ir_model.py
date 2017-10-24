#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
from scipy.interpolate import interp1d

import matplotlib
matplotlib.style.use('seaborn')

class Ir_pre(object):
    def __init__(self):
        self.ir = self.preprocessing()
        print self.ir

    def preprocessing(self, plot = False):
        ir = pd.read_csv('ir.csv', index_col = 0, parse_dates = True)
        ir = ir.replace(0.0, np.nan)
        ir = ir.fillna(method = 'pad')
        ir = ir.fillna(0.0)
        ir = ir.groupby(ir.index.strftime('%Y-%m')).last()
        ir.index = [datetime(int(x[:4]),int(x[-2:]),monthrange(int(x[:4]),int(x[-2:]))[1],) for x in ir.index]
        ir = ir.loc['2005-12':'2017-10']
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
    #ir_pre.cal_wacc()
    #ir_pre.pl_interp()

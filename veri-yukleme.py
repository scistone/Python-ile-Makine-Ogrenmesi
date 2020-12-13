#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:26:12 2020

@author: deger
"""

## Kütüphaneler
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt

## Verilerin import edilmesi

#csv dosylarını okumak için pandas'ın read_csv metodunu çağırıyoruz.
#read_csv() metodunun ilk parametresi file_path olduğu için verileri çekeceğimiz dosyanın yolunu pandas'a gösteriyoruz.
veriler = pd.read_csv('./veriler.csv') 

#------------------Sütun isimleri-------------------------------------------------
"""
verileri bu şekilde okuttuğumuzda eğer başka bir header değeri atamadıysak, 
pandas her zaman en üstteki satırı sütun isimleri olarak görür
"""
#-------------------pandas.read_csv() metodunun parametreleri:-------------------------------------
"""
pandas.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', 
names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, 
dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, 
skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, 
skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, 
dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', 
lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None,
dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=True, memory_map=False, 
float_precision=None)
"""
#-------------------------------------------------------------------


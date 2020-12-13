# Python-ile-Makine-Ogrenmesi
Python ile Makine Öğrenmesi Türkçe Döküman

# Kurulumlar

Öncelikle makine öğrenmesi projeleri geliştirirken benim kişisel tercihim olan Anaconda ve Anaconda'nın içerisinde gelen Spyder editörünü kullanmaktır.

## İhtiyacımız olan Kütüphaneler

1. Pandas
2. Numpy
3. Matplotlib




# Metodoloji : CRISP-DM

Bu dökümanda başlarken kullanacağımız yöntem Cross Industry Standard Process for Data Mining(CRISP-DM) metodolojisidir. 

Bu metodoloji bir hayat döngüsü şeklinde verinin etrafında dönen bir süreçtir.

Sırasıyla CRISP-DM hayat döngüsü:
1. Veriyi anlamak
2. Veri Ön İşleme
3. Modelleme - Kullanılacak yöntemin hayata geçirilmesi(Tahmin,Sınıflandırma,Bölütleme,Pekiştirmeli Öğrenme vb.)
4. Değerlendirme
    - İş Anlayışı-İşi anlama
5. Dağıtım

## Veri Ön İşleme(Data Preprocessing)

### Verilerin import edilmesi

```python
## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Verilerin import edilmesi
veriler = pd.read_csv('./veriler.csv') #csv dosylarını okumak için pandas'ın read_csv metodunu çağırıyoruz.


"""
pandas.read_csv() metodunun parametreleri: 

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

```
### Eksik veriler

Şunu unutmayalım ki, veriler her zaman düzgün bir şekilde olmak zorunda değillerdir. Ve eksik veri kavramı gündelik hayatta makine öğrenmesi projeleri geliştirirken sık karşılaşılan bir durumdur. Bu sebeple okuduğumuz verilerdeki eksik verileri veri ön işleme aşamasında manipüle etmemiz gerekmektedir. Bu değerler pandas'da NaN(Not a Number) şeklinde gösterilmektedir. 

Eksik veriler manipüle edilirken verinin tipine bağlı olarak farklı metodlarla manipüle edilebilir.

Bizim örneğimizde `eksik-veriler.csv` dosyamızda 12. ve 16. satırdaki yaş verilerimiz, verimizi temin ederken eksik gelmiştir. Bu sebeple bu değer bir yaş değeri olduğu için ve analizimizde bizi her hangi bir yanılgıya  uğratmaması için en basit yöntem olan tüm yaşların ortalamasını alarak, eksik değerlerin yerine işleyeceğiz.

```python
## Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

## Verilerin import edilmesi
veriler = pd.read_csv('./eksik-veriler.csv') 

# imputer ayarları burada yapılıyor
# missing values olarak nan olan değerleri gösteriyoruz
# strategy olarak da mean yani ortalama olduğunu söylüyoruz
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

#yaş ile alakalı olan sütunu seçiyoruz
age = veriler.iloc[:,3:4].values 

"""
iloc ana olarak iki adet parametre almaktadır. 
birincisi hangi satırdan hangi satıra kadar veriyi okuyacağı
ikincisi  hangi sütundan hangi sütuna kadar veriyi okuyacağı
"""

# age olarak aldığımız yaş sütununu imputer'a gönderip ilk belirlediğimiz stratejiye göre öğrenme işlemini gerçekleştiriyoruz.
imputer = imputer.fit(age)

# daha sonrasında yaş sütunu için imputer.transform() metodunu çağırıp değerlerini değiştiriyoruz..
age =  imputer.transform(age)
```






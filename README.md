# Python-ile-Makine-Ogrenmesi
Python ile Makine Öğrenmesi Türkçe Döküman. Sadi Evren Şeker'in Python ile Makine Öğrenmesi dersinden alıntı yapılmıştır. Kendi öğrenme sürecimde aldığım notlardır.

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

# Veri Ön İşleme(Data Preprocessing)

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

## Veri 


Aslında veri iki ana grupta incelenebilir. Kategorik veriler ve sayısal veriler. Örneğin; cinsiyet verisi, eğitim durumu, plaka numaraları, ülke verileri gibi veriler kategorik verilerdir. Bu gibi verilerde sayısal, büyüklük küçüklük ilişkisi kurulamayan verilerdir. 


Aslında bizim amacımız kategorik olan verileri de sayısal bir değere dönüştürüp(encoding) bu şekilde makine öğrenmesi algoritmasını çalıştırabilmektir.


- Veri
    - Kategorik Veriler
        - Nominal
        - Ordinal
    - Sayısal Veriler
        - Oransal (Ratio)
        - Aralık (Interval)

### Kategorik Veriler 

Ordinal veriler sıraya sokulabilen büyüktür küçüktür ilişkisi kurulabilen verilerdir. 

Nominal veriler ise sıralama ve büyüktür küçüktür ilişkisi kurulamayan verilerdir.

### Sayısal Veriler

Oransal(Ratio) birbirine göre orantılanabilen, çarpılıp bölünebilen değerlerdir.

Aralık ise birbiri ile çarpılıp bölünemeyen değerlerdir. Belli bir aralığı belli eden sayısal değerlerdir.

### Örnek

Örneğin elimizde ülke, boy, kilo, yaş ve cinsiyet verileri olan bir veri setimiz olsun. Problemimiz de ülke, boy, kilo ve yaş verileri ile cinsiyet bulmak olsun. Burada ülke değeri bir kategorik veridir ve burada bu veriyi kullanmak için sayısal bir değere çevirmeliyiz ki makine öğrenmesi algoritmamızı çalıştırabilelim.

```
Bu bölümle ilgili Python kodları veri-on-isleme klasöründe bulunmaktadır.
```

# Tahmin(Prediction)

Veri ön işleme kısmında veri başlığımız altında sayısal ve kategorik verilere değinmiştik. Aslında bu ayrım bize problem tiplerini ayırmamıza da yardım ediyor. Kategorik veriler üzerinde herhangi bir tahmin yapıldığında sınıflandırma(classification) problemi, sayısal veriler üzerinde tahmin yapıldığında tahmin(prediction) problemi oluyor.

Örnek olarak; 
- Bir kişinin yaşını, gelir düzeyini, dolar kurunu gibi sayısal verileri tahmin etmek istiyor isek bunu tahmin(prediction) olarak nitelendiriyoruz. 
- Bir kişinin eğitim düzeyini, cinsiyetini gibi kategorik verileri tahmin etmeye çalışıyorsak sınıflandırma olarak nitelendiriyoruz.

### Tahmin(Prediction) ve Öngörü(Forecasting)

Tahmin daha genel kapsayıcı bir kavramdır, öngörü ise daha özel bir kavramdır. Öngörü geleceğin tahmin edilmesine denmektedir. Yani öngörü bizim örneklem uzayımızın dışındaki olayları tahmin edilmesine denmektedir.

Tahminde geçmiş ya da gelecek ile bağıntı kurması önemli değildir. Eksik bir verinin tahmini de tahmin olarak nitelendirilir.

## Doğrusal Regresyon(Linear Regression)

Aslında burada amacımız veriler ile en iyi doğruyu yaratabilmektir.Yani bu doğruya en yakın geçen noktaları bularak bir doğru modeli inşaa etmektir. Aynı zamanda amacımız bu doğruda hata miktarımızı minimize etmektir. 

Örneğin aylara göre satış miktarı olan bir veride bir sonraki ay yapılacak satışları bulmak için doğrusal regresyon kullanabiliriz. 


### Basit Doğrusal Regresyon(Simple Linear Regression)

İki boyutlu bir uzayda basit bir doğruyu ifade etmek istersek matematiksel olarak:

y = ax+b + e olarak ifade edebiliriz.

Burada y değişkeni bağımlı değişken, x değeri bağımsız değişkendir. a değeri katsayı(coefficient) veya doğrunun eğimidir. b değeri ise sabit değerimizdir. e değeri hata miktarıdır.

### Çoklu Doğrusal Regresyon(Multiple Linear Regression)

Çoklu doğrusal regresyon, bağımlı değişkenin birden fazla olduğu durumlarda kullanılmaktadır.

### İstatistik Değerlerini Bulma

Kuracağımız makine öğrenmesi modeline değişkenlerin etkisini bulmak için aşağıdaki kodu kullanabiliriz.

```python
import statsmodels.api as sm

new_framelist = dataset.iloc[:,[0,1,2,3,4,5]].values
new_framelist = np.array(new_framelist,dtype=float)


model = sm.OLS(height,new_framelist).fit()
summary = model.summary()
```


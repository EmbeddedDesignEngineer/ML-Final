# Qiymət səhvlərinin daha dərin yoxlanması

Bu tədqiqat əvvəlki 24 ağaclı modeldən sonra qalan səhvləri araşdırır.
Train/validation bölgüsü dəyişmir. Test bölməsi qiymətləndirilmir. Validation
artıq bir neçə seçimdə istifadə edildiyindən nəticələr müstəqil benchmark deyil.
Target elan qiymətidir; bağlanmış alqı-satqının faktiki qiyməti kimi yoxlanılmayıb.

## Ölçülmüş nəticə

| Validation ölçüsü | Əvvəlki bagging | Dərin log-m² bagging |
|---|---:|---:|
| RMSE | 105,709.86 AZN | **98,742.16 AZN** |
| MAE | 40,217.07 AZN | **35,044.54 AZN** |
| R² | 0.7898 | **0.8166** |
| Median mütləq səhv | 17,684.54 AZN | **13,538.35 AZN** |
| 90-cı persentil mütləq səhv | 81,142.97 AZN | **72,862.08 AZN** |
| Median nisbi səhv | 8.72% | **6.75%** |
| ±10% daxilində | 55.37% | **63.40%** |
| ±20% daxilində | 79.99% | **84.27%** |

RMSE 6.6%, MAE 12.9% azalır. Bu, bu grid-də RMSE və MAE üçün qalibdir;
qlobal ən yaxşı model iddiası deyil. Raw-target dərin bagging median nisbi
səhvdə daha yaxşıdır (6.48%), amma RMSE=102,882, MAE=36,115 AZN ilə qalır.
Məqsəd seçimi buna görə açıq saxlanır.

Seçilən model 24 ağac, depth=24, leaf=1, bütün sütunlar, bootstrap sample=0.8,
seed=42 və log-price-per-area target istifadə edir. Böyük səhvləri azaldır,
amma xüsusilə bahalı evlərdə etibarlı qiymətləndirmə problemi bitmir.
Yeni modeldə ən pis 1% sətrin kvadrat səhvdə payı hələ **73.2%**-dir.

| Faktiki qiymət | Yeni MAE | Yeni ±20% daxilində |
|---|---:|---:|
| 100 min AZN-dən az | 14,966 AZN | 69.2% |
| 100–200 min AZN | 14,674 AZN | 89.2% |
| 200–500 min AZN | 32,718 AZN | 86.4% |
| 500 min AZN və yuxarı | 169,991 AZN | 66.4% |

Ən bahalı qrupda raw-target dərin bagging-in MAE-si 167,235 AZN ilə azacıq
aşağıdır. Faktiki qiymətə görə model seçməklə bu üstünlüklər birləşdirilmir:
bu, proqnoz zamanı bilinməyən target-dən istifadə etmək olardı.

## Yoxlanan ehtimallar

1. Hər ağacda sütunların 20%-nin çıxarılması sahə və məkan kimi vacib
   məlumatları bəzən kənarda qoyur. Bütün sütunlarla bagging ayrıca yoxlanır.
2. `log(price / area)` böyük və kiçik evlərin qiymət miqyasını yaxınlaşdırır.
   Proqnoz `exp(model(X)) * area` ilə AZN-ə qaytarılır. Sahə girişdən gəlir;
   validation qiymətindən feature düzəldilmir. Bu transformasiya RMSE-ni
   mütləq yaxşılaşdırmalı deyil və qərəzsiz conditional mean iddiası daşımır.
3. Daha dərin ağaclar və kiçik yarpaqlar bagging içində daha detallı əlaqə
   öyrənə bilər. Depth=24, leaf=1 raw və log-area target-ları ilə yoxlanır.
4. Residual boosting hər mərhələdə əvvəlki modelin səhvinə öz NumPy ağacını
   fit edir. 100 ağac, depth=5, learning_rate=0.1 raw və log-area ilə yoxlanır.
   Daha mürəkkəb model avtomatik qalib sayılmır.

Sklearn yalnız ayrıca ilkin müqayisə üçün istifadə edildi: eyni train/validation
üzərində HistGradientBoosting (300 iterasiya, 31 yarpaq, L2=10, early_stopping=False,
seed=42) raw target ilə RMSE=102,823, MAE=41,796 AZN verdi. Log-price ilə
RMSE=105,667, MAE=39,260 oldu. Bu iki nəticə ilkin araşdırmadır; custom
modellərin final seçim siyahısına daxil deyil. Standart kitabxana modelində də
böyük səhvin qalması problemi təkcə öz ağac implementasiyamıza bağlamağa əsas vermir.

Mənbələr: [Sklearn gradient boosting sənədi](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
kvadrat və mütləq səhv məqsədlərini ayırır;
[ensemble izahı](https://scikit-learn.org/stable/modules/ensemble.html)
boosting-in ardıcıl səhv düzəltmə prinsipini izah edir. Implementasiya
öz `DecisionTree` sinfimizdən istifadə edir.

## Böyük orta səhvin mənbəyi

Train medianını bütün evlərə proqnoz edən sadə baza modelinin MAE-si
123,781 AZN, RMSE-si 236,676 AZN-dir. Train mean bazasında RMSE=230,561 AZN.
Beləliklə, əvvəlki model də baza səviyyəsindən xeyli yaxşıdır, amma bu
real istifadə üçün xətanın avtomatik qəbuledilən olduğu demək deyil.

Əvvəlki bagging modelində ən böyük səhvə sahib validation sətirlərinin 1%-i
ümumi kvadrat səhvin **70.6%-ni** yaradır. Bu sətirlər silinməyib və final
ölçüdən çıxarılmayıb. Yalnız uğurlu qiymət aralığını göstərərək nəticə
yaxşılaşdırılmır.

| Faktiki qiymət | Say | Əvvəlki MAE | ±20% daxilində |
|---|---:|---:|---:|
| 100 min AZN-dən az | 814 | 21,622 AZN | 57.1% |
| 100–200 min AZN | 3,301 | 18,461 AZN | 84.3% |
| 200–500 min AZN | 3,804 | 37,299 AZN | 84.3% |
| 500 min AZN və yuxarı | 685 | 183,362 AZN | 62.3% |

Qruplar faktiki qiymətlə qurulub və yalnız diaqnostika üçündür. İnferens zamanı
qiymət bilinmədiyindən onlarla model yönləndirməsi edilmir. Dar qiymət
aralıqlarında R²-nin aşağı/mənfi olması həmin aralığın kiçik target variasiyası
ilə də bağlıdır; MAE və nisbi səhvlərlə birlikdə oxunmalıdır.

Train məlumatında 40,155 sətrə 39,798 fərqli feature vektoru düşür. Eyni
feature-lərlə fərqli qiymət olan 115 qrup var; 28 qrupda fərq 20%-dən böyükdür.
Bu təkbaşına bütün səhvi izah etmir və sübut olunmuş noise floor deyil.
Mövcud sütunların tam ayıra bilmədiyi elanlar olduğunu göstərir. Dəqiq ünvan,
təmirin keyfiyyəti, mənzərə və real satış qiyməti kimi əlavə etibarlı məlumatların
faydası ayrıca dataset işi tələb edir; bu sütunlar mövcudmuş kimi uydurulmayıb.

## Təkrarlamaq

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m src.run_all --price-study --clean data/house_sale_clean.csv --output results/price-new
```

Nəticələr: `errors.json`, `selected.json`, `manifest.json`. Mean və median
baza modelləri yalnız train target-lərindən qurulur. Seçilən adların parametrləri
manifestdədir. Sahə müsbət olmalıdır. Log-area wrapper proqnozları AZN-ə qaytarır.
Yoxlanmış tam run `artifacts/price_quality/` qovluğundadır: 7 model namizədi
və 2 baza modelinin nəticələri tamamlanıb, source hash-ləri yoxlanılıb.
Seçilən modelin ayrıca təkrar sınağı ilə bütün error-report rəqəmləri eynidir.

Seçilən modeli birbaşa istifadə etmək:

```python
from src.ensemble import BaggedTreeRegressor
from src.price_models import PriceRegressor

model = PriceRegressor(
    BaggedTreeRegressor(n_estimators=24, max_depth=24, min_samples_leaf=1,
                        max_features=1.0, max_samples=0.8, random_state=42),
    target="log_price_per_area",
    area_index=splits["feature_names"].index("area_m2"),
)
model.fit(splits["X_tr"], splits["y_tr"])
predictions_azn = model.predict(splits["X_val"])
```

Bu girişlər `get_splits()`-in ağaclar üçün verdiyi standardize edilməmiş
feature-lərdir; sahə sütununu standardize etmək olmaz. Əvvəlki full pipeline
tarixi modelləri saxlayır, yeni variant `--price-study` və bu model API-si ilə
istifadə olunur. Təlim əvvəlki dayaz bagging-dən bahadır; fit müddətləri
`errors.json`-da saxlanır və konkret mühitdən asılıdır.

Kod yoxlaması: **343 test keçdi**. Integer qiymətlərdə kvadrat səhvin overflow-u
və uğursuz refit-in əvvəlki boosting modelini dəyişməsi kod baxışında tapıldı,
düzəldildi və ayrıca testlə qorundu. Köhnə nəticə qovluğunun əzilməsi rədd edilir.

Boosting ayrıca sintetik datada sklearn ilə müstəqil yoxlanıldı: 3 depth və
3 learning-rate kombinasiyasında bütün 12 mərhələnin ən böyük fərqi
5.69e-14 oldu. Target transformasiyalarının tərs çevrilməsi də uyğun gəldi.
Real datada zəif boosting nəticəsini riyazi implementasiya xətası kimi
təqdim etməyə bu yoxlamada əsas tapılmadı.

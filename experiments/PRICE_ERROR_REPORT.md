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
İlk yoxlanmış run `artifacts/price_quality/` qovluğundadır (7 namizəd).
`deep48_log_area_bagging` əlavə edildikdən sonra 8 namizədli təkrar run
(`artifacts/price_quality_v2/`) əvvəlki 7 namizədin bütün rəqəmlərini eyni verdi və yeni namizədi RMSE, MAE,
median nisbi səhv üzrə seçdi.

Seçilən modeli birbaşa istifadə etmək:

```python
from src.ensemble import BaggedTreeRegressor
from src.price_models import PriceRegressor

model = PriceRegressor(
    BaggedTreeRegressor(n_estimators=48, max_depth=32, min_samples_leaf=1,
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

Kod yoxlaması: **348 test keçdi**. Integer qiymətlərdə kvadrat səhvin overflow-u
və uğursuz refit-in əvvəlki boosting modelini dəyişməsi kod baxışında tapıldı,
düzəldildi və ayrıca testlə qorundu. Köhnə nəticə qovluğunun əzilməsi rədd edilir.

Boosting ayrıca sintetik datada sklearn ilə müstəqil yoxlanıldı: 3 depth və
3 learning-rate kombinasiyasında bütün 12 mərhələnin ən böyük fərqi
5.69e-14 oldu. Target transformasiyalarının tərs çevrilməsi də uyğun gəldi.
Real datada zəif boosting nəticəsini riyazi implementasiya xətası kimi
təqdim etməyə bu yoxlamada əsas tapılmadı.

## Outlier, regularization və overfitting yoxlaması

```powershell
python -m src.run_all --price-diagnostics --output results/price-diagnostics-new
```

Bu komanda bütün aşağıdakı rəqəmləri yaradır (16 nüvəli maşında 12 proseslə
≈17 dəqiqə). Seçimlər rəsmi validation-a baxmadan edilib: rəsmi train hissəsi
tam dublikatlardan təmizlənir (bütün datada 446 təkrar sətir, rəsmi train-də
221), sonra 3 fərqli seed ilə (11, 23, 37) 80/20 daxili bölgüyə ayrılır. Hər
bölgüdə encoder yalnız daxili train-ə fit olunur; bagging 8 ağac, log-m² target
ilə ölçülür. Rəsmi validation yalnız seçilmiş modelin error auditində istifadə
olunur, test bölməsi ölçülmür. Hər fit ayrıca seed-lidir, ona görə paralel run
ardıcıl run ilə eyni rəqəmləri verir (testlə yoxlanılır). Nəticə faylları:
`regularization_cv.json`, `neighbour_cv.json`, `overfitting.json`,
`single_tree_grid.json`, `validation_audit.json`, `manifest.json`.
Kod: `experiments/price_diagnostics/`. Yoxlanmış run: `artifacts/price_diagnostics/`;
validation auditində elanların məkan və koordinat sahələri çıxarılıb.

### Outlier-lər

- Ən pis 30 validation səhvinin əksəriyyəti real görünən bahalı evlərdir
  (Bilgəh, Badamdar, Səbail; 1.3–4.2 mln AZN, m² qiyməti 3–8 min AZN).
  Mövcud sütunlarda (sahə, otaq, rayon) dəbdəbəni ayıran məlumat yoxdur.
- Bir neçəsi ehtimal ki data səhvidir: məsələn 600 m², 2 otaqlı köhnə tikili
  159 min AZN-ə (265 AZN/m²). Train-dən hesablanmış kateqoriya üzrə log-m²
  Q1−3·IQR … Q3+3·IQR sərhədindən kənarda validation-da cəmi 19 sətir var;
  ən pis 30 səhvdən yalnız 4-ü.
- Train-də bu ekstremal sətirlərin (hər bölgüdə 49) silinməsi daxili CV-də
  MAE-ni dəyişmədi (37,898 vs 38,027 eyni leaf=3 ilə), RMSE 0.2% azaldı.
  Fərq səs-küy səviyyəsindədir; silmə tətbiq edilmir. Validation heç vaxt
  dəyişdirilmir.
- Kalibrasiya: proqnoz qruplarında median faktiki/proqnoz nisbəti 0.992–1.003.
  Bahalı proqnozlarda sistematik aşağı qiymətləndirmə yoxdur; qalan səhv
  əsasən dağınıqlıqdır, bias deyil.
- Dublikat sızması: rəsmi validation-da 139 sətrin eyni feature-li əkizi
  train-dədir. Onlar çıxarıldıqda MAE 33,968 → 34,202 olur. Təsir kiçikdir,
  amma `data_prep.clean()` dublikatları yalnız URL üzrə silir; tam təkrar
  sətirlərin silinməsi komanda ilə razılaşdırılmalıdır.

### Bagging: regularization (3 daxili bölgü, orta)

| Konfiqurasiya | Train MAE | Val MAE | Val RMSE |
|---|---:|---:|---:|
| depth=24, leaf=1 | 19,506 | 37,902 | 108,902 |
| depth=24, leaf=3 | 23,250 | 38,027 | 107,223 |
| depth=16, leaf=5 | 31,139 | 40,149 | 110,031 |
| leaf=3 + ekstremal ppm silinməsi | 22,798 | 37,898 | 106,995 |

Güclü regularization (depth=16, leaf=5) val səhvini artırır, yəni bagging-də
problem underfit tərəfindədir.

### Bagging: overfitting əyriləri (2 daxili bölgü)

| depth / leaf | Train MAE | Val MAE (seed 11 / 23) | Val RMSE (seed 11 / 23) |
|---|---:|---:|---:|
| 6 / 1 | 50.0k | 51,330 / 51,165 | 121,374 / 121,363 |
| 10 / 1 | 40.6k | 45,932 / 44,189 | 116,314 / 105,133 |
| 14 / 1 | 31.4k | 42,423 / 40,543 | 113,686 / 102,417 |
| 18 / 1 | 24.5k | 40,131 / 38,597 | 112,001 / 100,986 |
| 24 / 1 | 19.5k | 39,043 / 37,247 | 111,253 / 100,143 |
| 32 / 1 | 18.4k | **38,826 / 37,113** | **111,197 / 100,103** |
| 24 / 3 | 23.3k | 39,134 / 37,310 | 109,663 / 100,324 |
| 24 / 10 | 31.6k | 40,244 / 39,297 | 111,003 / 106,387 |
| 24 / 30 | 38.9k | 43,236 / 42,525 | 115,517 / 114,243 |

- Train–val fərqi böyükdür (≈2 dəfə), amma val səhvi mürəkkəblik artdıqca
  hər iki bölgüdə **monoton azalır**, U-forması yoxdur. Bagging-də zərərli
  overfitting müşahidə edilmir: leaf=1 ilə tək ağaclar train-i demək olar
  əzbərləyir, bootstrap ağaclarının ortalaması isə variansı azaldır. Bu, sklearn RandomForest-in
  `max_depth=None`, `min_samples_leaf=1` default-una uyğundur.
- Learning curve (encoder tam daxili train-ə fit olunur): train-in
  25/50/75/100%-i ilə val MAE 44,587 → 41,967 → 40,470 → 38,723 (seed 11),
  45,164 → 41,308 → 39,380 → 38,169 (seed 23). Əyri doymayıb; əlavə real data
  ən etibarlı yaxşılaşma mənbəyidir.
- Ağac sayı (depth=24): 1 → 4 → 8 → 16 → 24 ağacda val MAE
  50,201 → 41,067 → 39,043 → 37,630 → 37,339 (seed 11). Ağac əlavə etmək
  val səhvini qaldırmır.

### Tək ağac (Task A, xam qiymət): depth × leaf

3 daxili bölgü, 9 depth × 6 leaf. Orta val RMSE (± bölgülər arası std):

| depth \ leaf | 1 | 5 | 20 | 100 |
|---|---:|---:|---:|---:|
| 8 | 125,721 | 125,198 | 127,498 | 131,850 |
| 12 | 129,613 | 125,850 ± 9,380 | 124,952 | 131,189 |
| 16 | 132,577 | 125,966 | **124,623 ± 734** | 131,158 |
| 24 | 134,339 | 126,205 | **124,554 ± 692** | 131,158 |
| limitsiz | 134,414 | 126,204 | 124,554 | 131,158 |

- **leaf=1 ilə tək ağac overfit edir:** depth 8-dən sonra val RMSE yenidən
  artır (125,721 → 134,414), train RMSE isə 93,592-dən 1,899-a düşür.
  Limitsiz ağac 44 dərinliyə və 25,128 yarpağa çatır.
- **leaf=20** ağacı özü dayandırır: depth ≥16 nəticəni dəyişmir, val RMSE ən
  aşağıdır və bölgülər arası std ±700-dür. leaf=5-də std ±9,400-dür: kiçik
  yarpaqlar bəzi bölgülərdə bir neçə ekstremal elanı əzbərləyir.
- leaf ≥50 underfit edir (val RMSE ≈129–131k).
- MAE fərqli göstərir: leaf=1-də depth artdıqca val MAE azalmağa davam edir
  (52.8k → 46.0k), RMSE isə artır; dərin ağac tipik elanları yaxşı, bir neçə
  bahalı evi isə çox səhv proqnoz edir.
- Final pipeline-dakı tək ağac (depth=12, leaf=5) dəyişdirilməyib. Bu grid
  əvvəlki sweep-dən (depth ≤12, leaf ≤20) genişdir və depth=16, leaf=20-nin
  ≈1% aşağı və daha sabit RMSE verdiyini göstərir.

### Rədd edilmiş: qonşu elanların m² qiyməti

Eyni kateqoriyada ən yaxın k train elanının median log-m² qiyməti
(train üçün 5-fold out-of-fold) əlavə feature kimi yoxlandı. 3 bölgü × 3
variantın hamısında pisləşdirdi: orta val MAE 40,157–41,180 (baza 37,902).
Model səs-küylü feature-ə həddən artıq güvənir. `src/`-ə daxil edilmədi;
kod `experiments/price_diagnostics/neighbour_feature.py`-dadır.

### Seçilmiş model: 48 ağac, depth=32

Daxili CV-nin göstərdiyi istiqamət (daha dərin, daha çox ağac) `--price-study`
namizədi `deep48_log_area_bagging` kimi əlavə edildi. Rəsmi validation-da
RMSE, MAE və median nisbi səhvin üçündə də qalibdir:

| Validation ölçüsü | 24 ağac, depth=24 | **48 ağac, depth=32** |
|---|---:|---:|
| RMSE | 98,742 AZN | **97,157 AZN** |
| MAE | 35,045 AZN | **33,968 AZN** |
| Median mütləq səhv | 13,538 AZN | **12,740 AZN** |
| R² | 0.8166 | **0.8224** |
| Median nisbi səhv | 6.75% | **6.41%** |
| ±10% daxilində | 63.40% | **64.32%** |
| ±20% daxilində | 84.27% | **84.45%** |
| 500 min+ qrupunda MAE | 169,991 AZN | **165,146 AZN** |

Bütün dörd qiymət qrupunda MAE azalır. Fit müddəti (`errors.json`) bu maşında ≈267 s-dən
≈672 s-ə qalxır. Ən pis 1%-in kvadrat səhvdə payı 73.7% qalır: bahalı evlər
problemi model tənzimləməsi ilə həll olunmur, əlavə feature (dəqiq ünvan,
təmir səviyyəsi, mənzərə) və ya daha çox data tələb edir. Validation artıq
çox seçimdə istifadə olunub; müstəqil qiymət test bölməsidir.

# Reqressiya və SVM yoxlaması — 17 oktyabr 2026

## Nəticə

Eyni dataset, seed=42 və eyni train/validation bölgüsü ilə ölçülüb.
57,363 təmiz sətrin 40,155-i train, 8,604-ü validation-dır.
Raw CSV SHA-256 əvvəlki eksperimentlə eynidir:
`b6a4c67e0d3c712de31b9db53db53688792bc03099a213c1994c43114758aec2`.
Data təmizləmədəki düzəlişlər bu dataset-in sətirlərini və sırasını dəyişmir.

| Model / validation ölçüsü | Əvvəl | Yeni |
|---|---:|---:|
| Reqressiya RMSE, AZN | 124,741.28 | 105,709.86 |
| Reqressiya MAE, AZN | 48,344.89 | 40,217.07 |
| Reqressiya R² | 0.7073 | 0.7898 |
| Linear SVM F1 | 0.8869 | 0.9071 |
| RFF-SVM F1 | 0.8407 | 0.8979 |

Reqressiyada RMSE təxminən 15.3% azalıb. Bu rəqəmlər validation nəticələridir;
əvvəlki test nəticələri ilə qarışdırılmamalıdır. Test bölməsi əvvəldən görüldüyünə
görə yeni, müstəqil yekun test nəticəsi iddia edilmir. Bütün sınanmış namizədlər
`artifacts/model_quality/validation.json` faylındadır; yalnız uğurlular seçilib göstərilməyib.

## Səbəblər və düzəlişlər

### Reqressiya

Əvvəlki ağac depth=12, leaf=5 idi. Depth=16, leaf=10 ilə tək ağacın
RMSE-si 118,296.20 AZN oldu. Bu, əvvəlki parametr axtarışının məhdud olduğunu
göstərir. Log-price sınağı AZN RMSE üzrə üstün çıxmadığı üçün avtomatik tətbiq edilmədi.

Əlavə model 24 öz NumPy ağacımızın proqnozunu ortalayır. Hər ağac train
sətirlərinin 80%-ni bootstrap ilə və sütunların 80%-ni təsadüfi seçir;
depth=16, leaf=5, seed=42. Bu, sabit ağacbaşına sütun seçimi olan bagging-dir;
hər node-da sütun seçən klassik random forest deyil. Sklearn ağacı istifadə edilmir.
Tək ağac tapşırığın əsas modeli olaraq qalır; ensemble əlavə müqayisədir.

RMSE hələ böyükdür: bahalı evlərdə böyük səhvlər kvadrat cəzanı artırır.
Bu düzəliş qiymət proqnozunun bütün obyektlər üçün dəqiq olması demək deyil.

### SVM

20,000 addım 40,155 train sətri üçün hətta bir stochastic epoch deyil.
Standart Pegasos-un train objective-i ilkin diaqnostikada 2.687 idi;
5 epoch-da 0.475-ə endi. Bu, kifayət qədər optimallaşdırılmadığını göstərir.

`average=True` son yarıdakı çəkiləri və bias-ı ortalayır. Objective və margin
göstəriciləri də qaytarılan modelə görə hesablanır. Əvvəlki davranış üçün default
`average=False` saxlanıb. Epoch burada əvəzləmə ilə n_train təsadüfi seçim deməkdir.

Linear model üçün seçilən parametr: lambda=1e-4, Pegasos, 20 epoch, averaging.
Constant schedule ilə F1=0.90710 çox yaxındır; Pegasos-un 0.90712 nəticəsi
praktik baxımdan aydın üstünlük sübut etmir.

RFF-də yalnız addımları artırmaq F1-i yaxşılaşdırmadı. Lambda=0.001 əvəzinə
1e-5, gamma=0.01, D=512, 20 epoch və averaging ilə F1=0.89792 oldu.
Gamma=0.1 bu bölgüdə pisləşdirdi. Parametrlər validation əsasında seçilib.
RFF nəticəsi linear modelin F1-indən hələ aşağıdır; qeyri-xətti modelin həmişə
üstün olması gözlənilməməlidir. Müxtəlif lambda-ların objective rəqəmləri
eyni optimallaşdırma məsələsinin müqayisəsi kimi təqdim edilmir.

### Digər tapılan səhvlər

- Mənfi rəqəmlər parsing zamanı müsbətə çevrilə bilirdi; işarə saxlanılır.
- URL-i olmayan fərqli elanlar dedup zamanı itə bilirdi; indi saxlanılır.
- Koordinatlar rəqəmə çevrilir; mənfi torpaq sahəsi rədd edilir.
- AZN olmayan açıq valyuta məlumatı səssiz qəbul edilmir.
- Yanlış split nisbətləri və tier threshold erkən yoxlanılır.
- Ridge normal tənlik matrisini quranda zəif şərtlənmiş sütunlarda sıradan çıxa
  bilirdi. SVD eyni regularizasiyanı daha dayanıqlı hesablayır; intercept cəzalanmır.
- Quality study mövcud nəticə qovluğunu əzmir. Manifest running/complete/failed
  statusu saxlayır; yarımçıq run tam müqayisə sayılmamalıdır.

## İstifadə və məhdudiyyətlər

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m src.run_all --quality-study --output results/quality-new
```

`selected.json` hər model ailəsi üçün seçilən constructor parametrlərini saxlayır.
SVM girişləri yalnız train statistikası ilə standardize edilməlidir.
Adi `python -m src.run_all` tarixi eksperiment axınını saxlayır; yeni müqayisə
üçün `--quality-study` lazımdır. Raw və hazırlanmış data Git-ə əlavə edilmir.

Bu, bir seed və bir validation bölgüsü üzrə inkişaf tədqiqatıdır. İlk diaqnostika
əsasında daha dərin ağac və sonra RFF grid-i genişləndirilib; əvvəlcədən
qeydiyyata alınmış müstəqil benchmark deyil. Yeni modellər üçün nested CV və
ya yeni toxunulmamış data ilə yekun qiymətləndirmə aparılmayıb. Tarixi CV
rəqəmləri yeni modellərə aid deyil. Məqsəd sübut edilən səhvləri düzəltmək və
daha yaxşı namizədlər verməkdir, qlobal optimal nəticə iddia etmək deyil.

İki ayrı kod baxışı riyazi implementasiyada əlavə problem tapmadı. Hər ikisinin
tapdığı output/provenance problemi düzəldildi və regression testi əlavə olundu.
Testlər numeric parsing, split/tier validation, kollinear Ridge, seeded averaging,
bagging və test datasından asılı olmayan seçim davranışını yoxlayır.

Tam test nəticəsi: 333 passed (Python 3.14, requirements.txt mühiti). Son 17 namizədin nəticəsi tamamlanıb və source hash-ləri yoxlanılıb.

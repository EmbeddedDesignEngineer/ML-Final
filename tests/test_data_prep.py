import numpy as np
import pandas as pd
import pytest
from src import data_prep as d

@pytest.fixture
def raw():
    return pd.DataFrame([{'datetime_scrape_x':'2026-01-01','estate_rel_url_x':'/1','Kateqoriya':'Yeni tikili','city':'bakı','price':200000,'Sahə':'100 m²','Otaq sayı':3,'Mərtəbə':'2 / 9','Torpaq sahəsi':None,'Təmir':'var','Çıxarış':'var','İpoteka':'var','products_label':'Agentlik','vip':None,'featured':None,'location':'Yasamal','lat':40.4,'lng':49.8,'currency_x':'AZN'}])

def test_negative_area_is_rejected(raw):
    raw['Sahə']='-100 m²'
    assert d.clean(raw,verbose=False).empty

def test_numeric_string_coordinates(raw):
    expected=d.clean(raw,verbose=False)
    raw['lat']='40.4'; raw['lng']='49.8'
    pd.testing.assert_frame_equal(d.clean(raw,verbose=False),expected)

def test_unknown_currency_fails_explicitly(raw):
    raw['currency_x']='USD'
    with pytest.raises(ValueError,match='currency'):
        d.clean(raw,verbose=False)

def test_distinct_missing_url_rows_survive(raw):
    raw['estate_rel_url_x']=None
    other=raw.assign(price=210000)
    assert len(d.clean(pd.concat([raw,other],ignore_index=True),verbose=False))==2

@pytest.mark.parametrize('val,test',[(.8,.8),(-.1,.15),(np.nan,.15)])
def test_invalid_split_sizes_fail(val,test):
    with pytest.raises(ValueError):
        d._split_indices(np.arange(100),val,test)

def test_split_disjoint_and_exhaustive():
    parts=d._split_indices(np.arange(100))
    assert len(set(np.concatenate(parts)))==sum(map(len,parts))==100

def test_tier_rejects_invalid_prices():
    with pytest.raises(ValueError):
        d.make_tier_label([100,np.nan])

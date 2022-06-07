import pandas as pd
import requests
import urllib
import json
import re

df_nielsen = pd.read_csv('./emtek/nielsen1.csv')
df_acara = pd.read_csv('./emtek/lengkap.csv')

in_id_acara = 'RID_NIELSEN'
in_tm_acara = 'AIRING_TIME'
in_tg_acara = 'AIRING_DATE'
in_pdc_acara = 'NIELSEN_PROD_NAME'

in_id_nielsen = 'RID_NIELSEN'
in_tm_nielsen = 'AIRING_TIME'
in_tg_nielsen = 'AIRING_DATE'
in_pdc_nielsen = 'PRODUCT_NAME'

dict_nielsen = df_nielsen.to_dict()
dict_acara = df_acara.to_dict()

data = {
    'sumber' : dict_acara,
    'kolom_waktu_sumber' : in_tm_acara,
    'kolom_tanggal_sumber' : in_tg_acara,
    'kolom_id_sumber' : in_id_acara,
    'kolom_product_sumber' : in_pdc_acara,
    'tujuan' : dict_nielsen,
    'kolom_waktu_tujuan' : in_tm_nielsen,
    'kolom_tanggal_tujuan' :in_tg_nielsen,
    'kolom_id_tujuan' : in_id_nielsen,
    'kolom_product_tujuan' : in_pdc_nielsen
    }
# -----------------------------------------------------------------
try:
    # url = "http://127.0.0.1:5000/match"
    url = 'http://34.101.243.177:4000/match'
        
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json')
    req.add_header('User-Agent', 'Mozilla/5.0')
        
    jsondata = json.dumps(data)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
        
    print ("send: {}".format(jsondataasbytes))
        
    f = urllib.request.urlopen(req, jsondataasbytes, timeout=60)
    response = f.read()
        
    print("receive: {}".format(response))
    f.close()
    
except Exception as e:
    print(e)
    print("ignore exception, continue")

re0 = response.decode()
res = json.loads(re0)
df_hasil = pd.DataFrame.from_dict(res, orient='columns') 
print (df_hasil.head())

df_hasil.to_csv(r'./emtek/hasil1.csv', index=False)
# print (re0)

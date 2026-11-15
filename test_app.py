import json, sys, joblib, pandas as pd
sys.path.insert(0, '/home/claude/work/app')
from streamlit.testing.v1 import AppTest
R = json.load(open('/home/claude/work/resultados.json'))
OUT = {}
def nueva():
    at = AppTest.from_file('/home/claude/work/app/app.py', default_timeout=60); at.run(); return at
def cargar(at, e):
    for k, v in e.items():
        radios = {x.key: x for x in at.radio}
        if k == 'sex': radios['sex'].set_value('Masculino' if v == 1 else 'Femenino')
        elif k in radios: radios[k].set_value('Sí' if v == 1 else 'No')
        elif k in {x.key for x in at.selectbox}: at.selectbox(key=k).set_value(int(v))
        else: at.number_input(key=k).set_value(float(v) if k == 'bmi' else int(v))
# 1 entrada válida
at = nueva(); at.button(key='calcular').click().run()
OUT['valida'] = dict(excepciones=len(at.exception), prob=at.session_state['ultima_prob'], metricas=[m.value for m in at.metric])
# 2 valores límite
res = []
for bmi, mh, ph in [(12.0, 0, 0), (98.0, 30, 30)]:
    at = nueva(); at.number_input(key='bmi').set_value(bmi); at.number_input(key='menthlth').set_value(mh); at.number_input(key='physhlth').set_value(ph)
    at.button(key='calcular').click().run(); res.append(dict(bmi=bmi, mh=mh, ph=ph, excepciones=len(at.exception), prob=round(at.session_state['ultima_prob'], 4)))
OUT['limite'] = res
# 3 inválida: valor fuera de dominio en widget
at = nueva()
try:
    at.number_input(key='bmi').set_value(5.0).run(); OUT['fuera_rango_widget'] = dict(excepciones=len(at.exception), errores=[e.value for e in at.error], valor_bmi=at.number_input(key='bmi').value)
except Exception as ex:
    OUT['fuera_rango_widget'] = dict(error=type(ex).__name__ + ': ' + str(ex)[:150])
import app as A
OUT['validar_incompleto'] = A.validar({'bmi': 27})[:2]
d = {c: 0 for c in A.CAMPOS}; d.update(bmi=150, genhlth=9, age=14, highbp=2, menthlth=40)
OUT['validar_fuera_dominio'] = A.validar(d)
# 4 consistencia notebook vs app
modelo = joblib.load('/home/claude/work/modelo_diabetes.joblib'); cons = []
for caso in R['consistencia']:
    at = nueva(); cargar(at, caso['entrada']); at.button(key='calcular').click().run()
    pa = at.session_state['ultima_prob']; cons.append(dict(prob_notebook=round(caso['probabilidad'], 6), prob_app=round(pa, 6), clase_nb=caso['prediccion'], clase_app=at.session_state['ultima_clase'], dif=abs(pa-caso['probabilidad'])))
OUT['consistencia'] = cons
json.dump(OUT, open('/home/claude/work/resultados_app.json', 'w'), indent=1, default=str); print(json.dumps(OUT, indent=1, default=str))

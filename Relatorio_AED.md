# Análise Descritiva do Conjunto de Dados de Doença Cardíaca

Este documento apresenta uma análise descritiva das variáveis contidas no conjunto de dados sobre doenças cardíacas, compilado a partir de quatro fontes diferentes.

---

### **1. id (Identificador único para cada paciente)**
- Representa um identificador único de cada paciente.
- Não apresenta insights analíticos diretos além de sua função de indexação.

---

### **2. age (Idade do paciente em anos)**
- **Média:** 53.51 anos; **Mediana:** 54.00 anos.
- Distribuição majoritariamente masculina em todas as faixas etárias.
- **Variação por dataset:**
  - **VA Long Beach:** idade média mais alta (59.35 anos)
  - **Hungary:** idade média mais baixa (47.89 anos)
- Pacientes com angina atípica (*atypical angina*) são mais jovens (~49 anos).
- Correlação leve positiva com `trestbps` (0.24) e `oldpeak` (0.26).
- Correlação negativa com `thalch` (-0.37).
- Pacientes com angina induzida por exercício (`exang=True`) são mais velhos (~56 anos) que os sem (`exang=False`: ~52 anos).
- Idade média aumenta com a gravidade da doença cardíaca (`num`):
  - `num=0`: 50.55 anos
  - `num=3-4`: 59.21 anos

---

### **3. sex (Sexo do paciente)**
- **Masculino:** 726 pacientes; **Feminino:** 194 pacientes.
- Homens representam 274.23% a mais que mulheres.
- Predominância masculina significativa em todos os datasets, especialmente Switzerland e VA Long Beach.
- Homens têm maior `oldpeak` médio (0.94 vs. 0.67) e mediana (0.6 vs. 0.0) que mulheres.
- Pressão arterial em repouso (`trestbps`) semelhante entre os sexos.
- Homens são mais frequentes em todos os tipos de dor no peito (`cp`) e graus de doença cardíaca (`num`).

---

### **4. dataset (Origem geográfica ou fonte do estudo)**
- **Distribuição de registros:** Cleveland (304), Hungary (293), VA Long Beach (200), Switzerland (123).
- Idade média varia: VA Long Beach mais alta, Hungary mais baixa.
- **Switzerland:** todos os valores de `chol` são 0.0 (dados ausentes).
- **Cleveland:** maior contribuição para hipertrofia ventricular esquerda (`restecg`).
- **VA Long Beach:** maior média de `oldpeak` (1.32); **Hungary:** menor (0.59).

---

### **5. cp (Tipo de dor no peito)**
- **Frequência:** *asymptomatic* (496), *typical angina* (46) — menos frequente.
- *Asymptomatic* mais comum em todos os datasets, especialmente Cleveland e VA Long Beach.
- Pacientes com *atypical angina* são mais jovens (~49 anos).
- **`oldpeak` médio:**
  - *Asymptomatic*: 1.16
  - *Typical angina*: 1.07
  - *Atypical angina*: 0.30

---

### **6. trestbps (Pressão arterial em repouso)**
- Variável de 0.0 a 200.0 mm Hg; **média:** 132.13 mm Hg.
- Correlação leve positiva com idade (0.24).
- VA Long Beach apresenta as maiores médias de pressão: homens 140.0 mm Hg, mulheres 138.8 mm Hg; Hungary com menores médias.

---

### **7. chol (Colesterol sérico)**
- Variável de 0.0 a 603.0 mg/dL; **média:** 199.13 mg/dL.
- 172 valores zero (possíveis dados ausentes ou inválidos), 123 desses no dataset Switzerland.
- Correlação negativa com idade (-0.09).
- Pacientes com 3 vasos principais (`ca=3.0`) têm colesterol médio mais alto (265.45 mg/dL).
- Médias de colesterol menores nos graus mais avançados de doença cardíaca (`num=2-4`).

---

### **8. fbs (Glicemia de jejum > 120 mg/dL)**
- **False** (≤120 mg/dL): 692 pacientes; **True** (>120 mg/dL): 138 pacientes.
- Pacientes com `fbs=True` tendem a ser mais velhos que os com `fbs=False`.

---

### **9. restecg (Resultados do ECG em repouso)**
- **Distribuição:** *normal* (551), *lv hypertrophy* (188), *st-t abnormality* (179).
- **Idade média:** *normal* ~51.76 anos; *st-t abnormality* e *lv hypertrophy* ~56 anos.
- Cleveland domina *lv hypertrophy*; Hungary domina *normal*.
- **Pressão arterial em repouso:** *st-t abnormality* (135.84 mm Hg), *lv hypertrophy* (134.27 mm Hg), *normal* (130.38 mm Hg).

---

### **10. thalch (Frequência cardíaca máxima atingida)**
- Variável de 60.0 a 202.0 bpm; **média:** 137.55 bpm; **mediana:** 140 bpm.
- Correlação negativa com idade (-0.37).
- Pacientes sem angina induzida (`exang=False`) ~145 bpm; com angina (`exang=True`) ~126 bpm.

---

### **11. exang (Angina induzida por exercício)**
- **False:** 528; **True:** 337.
- Pacientes com angina induzida (True) são mais velhos (~55.5 anos) e têm menor `thalch` (~126 bpm) que os sem angina (~51.6 anos, ~145 bpm).
- Proporção de `exang=True` aumenta com a gravidade da doença (`num`): 14.07% em `num=0` até 68.09% em `num=3`.

---

### **12. oldpeak (Depressão do segmento ST)**
- Variável de -2.60 a 6.20; **média:** 0.88; **mediana:** 0.50.
- Homens têm valores médios maiores que mulheres.
- **VA Long Beach:** maior média (1.32); **Hungary:** menor (0.59).
- **Médias de Oldpeak:** *asymptomatic* 1.16, *typical angina* 1.07, *atypical angina* 0.30.
- Pacientes com `exang=True`: 1.41; com `exang=False`: 0.54.
- Correlação positiva com idade (0.26); negativa com `thalch` (-0.15).
- Aumenta com grau de doença cardíaca (`num`).

---

### **13. slope (Inclinação do segmento ST no pico do exercício)**
- **Distribuição:** *flat* (345), *upsloping* (203), *downsloping* (63); 309 valores nulos.
- *Flat* predominante em todos os sexos e tipos de dor no peito.
- *Downsloping* associado a idade ligeiramente maior.
- *Flat* prevalente em todos os graus de doença cardíaca (`num`).

---

### **14. ca (Número de principais vasos)**
- **Distribuição:** 0 vasos (181), 1 vaso (67), 2 vasos (41), 3 vasos (20); 611 valores nulos.
- Pacientes com 3 vasos (`ca=3.0`) têm colesterol médio mais alto (265.45 mg/dL) e maior Oldpeak médio (1.86) que pacientes com 0 vasos (0.83).

---

### **15. thal (Thalassemia)**
- **Distribuição:** *normal* (196), *reversable defect* (192), *fixed defect* (46); 486 valores nulos.
- *Normal* predominante no dataset Cleveland; *reversable defect* notável em Switzerland.
- **Idade média:** *reversable defect* 55.92 anos; *normal* 53.20 anos.
- Distribuição clara por grau de doença (`num`): *normal* mais comum em `num=0`; *reversable/fixed defect* mais frequentes em graus elevados.

---

### **16. num (Presença/ausência de doença cardíaca - variável alvo)**
- **Distribuição:** 0 (411), 1 (265), 2 (109), 3 (107), 4 (28).
- Idade média aumenta com gravidade: de 50.55 anos (`num=0`) a 59.21 anos (`num=3-4`).
- Predominância masculina em todos os graus.
- Cleveland: maior proporção de `num=0`; Hungary e VA Long Beach contribuem mais para graus de doença.
- Colesterol médio tende a ser menor nos graus mais avançados.
- Percentual de `exang=True` maior nos graus mais elevados.
- `Oldpeak` aumenta com o grau de doença cardíaca.
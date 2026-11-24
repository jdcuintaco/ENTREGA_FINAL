# Anonimización de historias clínicas colombianas con BETO

Proyecto final del curso de Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana.

**Integrantes:** Juan David Cuintaco, Santiago Serrano, Juan Felipe Pardo Quintana y Óscar Iván García Montañez.

## 1. Descripción breve

Las historias clínicas están llenas de datos personales: nombres, cédulas, teléfonos, direcciones, fechas y registros médicos. Antes de usarlas para investigación, auditoría o analítica, una institución de salud debe retirar esos datos, y hacerlo a mano es lento y deja errores. Este proyecto construye un sistema que encuentra esos datos en texto clínico en español y los reemplaza por etiquetas genéricas como `[NOMBRE]` o `[IDENTIFICACIÓN]`.

El modelo base es BETO (`dccuchile/bert-base-spanish-wwm-cased`) afinado con el corpus MEDDOCAN (1.000 casos clínicos sintéticos de España, 22 tipos de dato personal). Como las notas colombianas escriben distinto la cédula, el celular, la dirección o el registro médico, el sistema final agrega tres capas para Colombia: copias colombianas del corpus para reajustar el modelo, reglas para formatos fijos y un postproceso que corrige el tipo por contexto, propaga lo detectado y baja el umbral de decisión para favorecer el recall.

**Tarea analítica de PLN.** Reconocimiento de entidades nombradas (NER) planteado como etiquetado de secuencias: cada subpalabra recibe una de 45 etiquetas BIO (`O` más `B-` e `I-` para cada uno de los 22 tipos de MEDDOCAN). Las entidades se reconstruyen a nivel de carácter y se sustituyen en el texto original.

**Usuario final.** El coordinador de historias clínicas o el analista de datos de una IPS, un hospital o una EPS en Colombia que debe entregar notas clínicas a un tercero (un grupo de investigación, un auditor, un equipo de analítica) cumpliendo la Ley 1581 de 2012 de protección de datos personales. Para ese usuario el error que importa es dejar un dato visible, por eso la métrica principal es la **fuga crítica**: el porcentaje de datos identificadores (nombres, documentos, teléfonos, direcciones, registros) que quedan sin tapar. El sistema apoya la revisión humana; no la reemplaza.

## 2. Resultados principales

Cifras de la última ejecución guardada en `notebooks/01_entrenamiento_y_exportacion.ipynb` (Colab, GPU A100, una sola semilla). La prueba colombiana son 80 notas sintéticas con 973 entidades, 460 de ellas críticas.

| Sistema | F1 MEDDOCAN (estricto) | F1 Colombia con tipo | F1 Colombia sin tipo | Fuga crítica | Notas sin fuga |
|---|---|---|---|---|---|
| Reglas (línea base, solo España) | 0,585 | n. d. | n. d. | n. d. | n. d. |
| BETO entrenado en España | 0,959 | 0,597 | 0,854 | 17,6 % | 32,5 % |
| BETO España + reglas CO | 0,959 | 0,666 | 0,902 | 2,0 % | 88,8 % |
| BETO adaptado + reglas CO | 0,966 | 0,696 | 0,885 | 3,5 % | 81,3 % |
| **BETO adaptado + reglas + mejoras (sistema exportado)** | **0,962** | **0,715** | **0,871** | **2,2 %** | **88,8 %** |
| RoBERTa clínico + reglas + mejoras | 0,951 | 0,663 | 0,819 | 2,4 % | 87,5 % |

El modelo de España deja visible el 17,6 % de los datos críticos en notas colombianas. Las reglas colombianas explican la mayor parte de la mejora, y el sistema final queda en 2,2 % de fuga (10 de 460 datos, todos nombres de pacientes) sin perder desempeño en MEDDOCAN. La prueba colombiana es sintética y la escribió el mismo equipo que escribió las reglas, así que estas cifras son un límite optimista. El informe en `docs/` discute las limitaciones.

## 3. Pipeline PLN

El notebook de entrenamiento está organizado en cajas. Cada una recibe una entrada, la transforma y entrega la salida que usa la siguiente.

| Caja | Etapa | Entra | Sale |
|---|---|---|---|
| 0 | Preparación | Repositorio o Colab vacío | Datos de MEDDOCAN en `data/raw/`, rutas, semilla y Drive montado |
| 1 | Adquisición, exploración y limpieza | `02_documentos.csv`, `01_anotaciones.csv` | 1.000 documentos con 22.795 entidades y offsets verificados (0 errores) |
| 2 | Etiquetado | Spans de carácter con su tipo | 45 etiquetas BIO por subpalabra |
| 3 | Seccionado y representación | Documentos completos | Ventanas de 512 tokens con solape de 128 (1.021 / 538 / 501) |
| 4 | Línea base | Texto clínico | Spans detectados con expresiones regulares |
| 5 | Modelado | Ventanas etiquetadas | BETO afinado para clasificación de tokens (10 épocas) |
| 6 | Evaluación | Modelo y prueba oficial | F1 estricto y relajado por documento, por tipo y matriz de confusión |
| 7 | Instanciación | Texto nuevo | Texto anonimizado con 6 etiquetas legibles y medición de fuga |
| 8 | Adaptación a Colombia | Modelo de España | Prueba colombiana sintética, reglas CO, aumento de datos y BETO reajustado (3 épocas) |
| 9 | Mejoras y cambio de modelo | Modelo adaptado | Tipo por contexto, propagación, umbral elegido en validación y comparación con RoBERTa clínico |
| 10 | Exportación | Sistema final | Carpeta `anonimizador_beto_co` (modelo, reglas, configuración y resultados) en Google Drive |

Un ejemplo real de la transformación, tomado de la salida de la Caja 2:

```text
Texto:      "Nombre: Ernesto. NHC: 368503."
Tokens:     Nombre  :  Ernesto                     .  NH  ##C  :  36                      ##85                    ##03                    .
Etiquetas:  O       O  B-NOMBRE_SUJETO_ASISTENCIA  O  O   O    O  B-ID_SUJETO_ASISTENCIA  I-ID_SUJETO_ASISTENCIA  I-ID_SUJETO_ASISTENCIA  O
Resultado:  "Nombre: [NOMBRE]. NHC: [IDENTIFICACIÓN]."
```

La figura `docs/figuras/arquitectura_beto.svg` muestra cómo BETO convierte esos tokens en 45 probabilidades por token.

## 4. Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   ├── 01_entrenamiento_y_exportacion.ipynb   # pipeline completo, Cajas 0 a 10, con salidas
│   └── 02_demo_anonimizacion.ipynb            # carga el sistema exportado y anonimiza texto nuevo
└── docs/
    ├── informe_final.pdf                      # documento descriptivo según Ashmore et al. (2021)
    ├── presentacion_sustentacion.pptx         # diapositivas de la sustentación
    └── figuras/
        └── arquitectura_beto.svg
```

El repositorio no incluye datos ni pesos. El corpus MEDDOCAN se descarga solo desde el repositorio público del Plan TL (`PlanTL-GOB-ES/SPACCC_MEDDOCAN`) y el modelo entrenado (unos 440 MB) lo genera el notebook 1 en Google Drive.

## 5. Requerimientos

| Componente | Notebook 1 (entrenamiento) | Notebook 2 (demo) |
|---|---|---|
| Entorno | Google Colab con GPU (T4 o superior) | Colab o Python local; CPU es suficiente |
| Python | 3.10 o superior | 3.10 o superior |
| Espacio en Google Drive | Cerca de 1 GB (carpeta exportada y zip de 406 MB) | Lectura de la carpeta exportada |
| Internet | Para descargar MEDDOCAN, BETO y RoBERTa clínico | Solo para instalar paquetes |

Paquetes de Python (`requirements.txt`): `torch`, `transformers>=4.46`, `datasets>=3.0`, `accelerate>=1.0`, `seqeval`, `scikit-learn`, `pandas`, `numpy` y `matplotlib`. Colab ya trae `torch`, `pandas`, `numpy` y `matplotlib`; la primera celda del notebook 1 instala el resto.

## 6. Instalación paso a paso

### Opción A: Google Colab (recomendada)

1. Abra `notebooks/01_entrenamiento_y_exportacion.ipynb` en Colab (en GitHub, botón **Open in Colab**, o en Colab **Archivo → Abrir notebook → GitHub** y pegue la URL del repositorio).
2. Elija GPU en **Entorno de ejecución → Cambiar tipo de entorno**.
3. Ejecute la primera celda (`%pip install ...`). Si Colab pide reiniciar el entorno, hágalo y siga desde la Caja 0.
4. Ejecute la Caja 0. Si no encuentra `data/raw/`, clona `SPACCC_MEDDOCAN` y arma los dos CSV automáticamente. La celda de Drive pide permiso para montar `MyDrive`.
5. Ejecute las Cajas 1 a 10 en orden (**Entorno de ejecución → Ejecutar todas** también sirve). La Caja 10 solo necesita las Cajas 0 a 9.2; si no hay tiempo para entrenar RoBERTa (9.3 y 9.4), la exportación funciona igual.
6. Al terminar queda en Drive la carpeta `MyDrive/meddocan-anon/export/anonimizador_beto_co/` y el archivo `anonimizador_beto_co.zip`.

Para depurar sin GPU, cambie `MODO_RAPIDO = True` en la Caja 0: el flujo corre con 8 documentos y 1 época. Esos resultados no sirven para reportar.

### Opción B: entorno local

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook notebooks/
```

En local la celda de Drive no hace nada y la exportación queda en `export/anonimizador_beto_co/` dentro del proyecto. Entrenar sin GPU no es práctico; para la demo local basta con copiar la carpeta exportada (o el zip) a la raíz del repositorio.

## 7. Cómo correr la demo

1. Abra `notebooks/02_demo_anonimizacion.ipynb` en Colab (o local).
2. **Celda 1.** Instala lo que falte y busca el sistema exportado en este orden: `MyDrive/meddocan-anon/export/anonimizador_beto_co`, `/content/anonimizador_beto_co`, `export/anonimizador_beto_co` o el zip `anonimizador_beto_co.zip` subido a Colab o a la raíz del proyecto. Si no lo encuentra, monta Drive y vuelve a buscar. Muestra las métricas con las que se exportó el sistema.
3. **Celda 2.** Define `anonimizar()` con las mismas reglas, el mismo umbral y el mismo postproceso del entrenamiento; las reglas se leen de `reglas.json`, no se copian a mano.
4. **Celda 3.** Cambie `TEXTO` por una nota propia o elija uno de los ejemplos (`"urgencias"`, `"evolucion"`, `"remision"`) y ejecute. La celda muestra el texto original, la tabla de entidades detectadas y el texto anonimizado.

Fragmento de la salida guardada con el ejemplo `"urgencias"` (601 ms por nota en CPU):

```text
Paciente Luz Dary Cárdenas Ospina, 58 años, sexo F, CC 38.256.914, afiliada a Salud Total EPS.
Traída por su hijo Wilmer Andrés Ospina, celular 316 782 4410. Reside en Calle 60 # 5-32,

Paciente [NOMBRE] [NOMBRE] [NOMBRE][NOMBRE], [OTRO], [IDENTIFICACIÓN] F, [IDENTIFICACIÓN], afiliada a [LUGAR] [LUGAR].
Traída por su [NOMBRE], [CONTACTO]. Reside en [LUGAR], [LUGAR] ([LUGAR]). ...
```

Los datos críticos quedan tapados, pero la salida también muestra errores que siguen abiertos: el nombre de la paciente se parte en cuatro fragmentos con tipos mezclados ("Luz" sale como personal sanitario y "Cárdenas Os" queda separado de "pina"), la palabra "sexo" se marca como identificación mientras la "F" queda visible, "sede" se marca como calle y el registro médico (RM) como teléfono. Ninguno de estos errores expone un dato crítico, pero empobrecen el texto y justifican la revisión humana.

## 8. Limitaciones

- La prueba colombiana es sintética y no hay notas reales colombianas anotadas; el siguiente paso es validar con historias reales de una institución, con aval ético.
- Una sola semilla y sin intervalos de confianza: diferencias de uno o dos puntos entre sistemas no deben leerse como reales. Con 460 datos críticos, cada dato mueve la fuga 0,22 puntos.
- Los nombres de pacientes son el dato que más se escapa, porque no tienen una regla que los cubra.
- El sistema reemplaza por etiquetas; no genera seudónimos ni mide la reidentificación indirecta.

## 9. Referencias principales

- Ashmore, R., Calinescu, R. y Paterson, C. (2021). Assuring the machine learning lifecycle: Desiderata, methods, and challenges. *ACM Computing Surveys*, 54(5), artículo 111. https://doi.org/10.1145/3453444
- Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H. y Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. *PML4DC at ICLR 2020*.
- Marimon, M., Gonzalez-Agirre, A., Intxaurrondo, A., Rodríguez, H., Lopez Martin, J. A., Villegas, M. y Krallinger, M. (2019). Automatic de-identification of medical texts in Spanish: The MEDDOCAN track, corpus, guidelines, methods and evaluation of results. *IberLEF 2019*, CEUR-WS 2421, 618-638.

La lista completa está en `docs/informe_final.pdf`.

## 10. Licencia

El código y los notebooks se publican bajo licencia MIT (ver `LICENSE`). **Supuesto:** el equipo no ha confirmado la licencia; cámbiela si el curso o la universidad piden otra. El corpus MEDDOCAN y los modelos preentrenados (BETO, RoBERTa clínico del Plan TL) conservan sus propias licencias y no se redistribuyen en este repositorio.

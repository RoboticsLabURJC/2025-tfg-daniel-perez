# Índice del documento

## Capítulo 1. Introducción y objetivos
- 1.1 Contexto y motivación  
- 1.2 Objetivos  
- 1.3 Metodología  
- 1.4 Plan de trabajo  
- 1.5 Estructura del documento  

---

## Capítulo 2. Fundamentos y herramientas

### 2.1 Percepción en robótica autónoma  
- 2.1.1 Entornos estructurados vs no estructurados  
- 2.1.2 Robots autónomos en entornos forestales  

### 2.2 Sensores y representación del entorno  
- 2.2.1 Cámaras RGB  
- 2.2.2 Sensores LiDAR  
- 2.2.3 Nubes de puntos  

### 2.3 Segmentación del entorno  
- 2.3.1 Segmentación geométrica  
- 2.3.2 Segmentación semántica  

### 2.4 Alineamiento entre observaciones multimodales  
- 2.4.1 Calibración geométrica entre sensores  
- 2.4.2 Limitaciones de la calibración clásica  
- 2.4.3 Desfase temporal entre observaciones  
- 2.4.4 Alineamiento espacio-temporal entre observaciones  

---

## Capítulo 3. Simulación LiDAR y segmentación geométrica

### 3.1 Simulación LiDAR en CARLA  
- 3.1.1 Generación de datos sintéticos  
- 3.1.2 Configuración del sensor  

### 3.2 Simulación en condiciones adversas  
- 3.2.1 Simulación de humo  
- 3.2.2 Impacto en las nubes de puntos  

### 3.3 Segmentación geométrica  
- 3.3.1 Métodos basados en geometría  
- 3.3.2 Limitaciones  

### 3.4 Análisis de datos sintéticos  
- 3.4.1 Evaluación cualitativa  
- 3.4.2 Problemas detectados  

---

## Capítulo 4. Segmentación semántica y alineamiento multimodal

### 4.1 Segmentación semántica con Deep Learning  
- 4.1.1 Redes para imágenes (2D)  
- 4.1.2 Redes para nubes de puntos (3D)  
- 4.1.3 Modelos utilizados en el trabajo  

### 4.2 Aplicación a datasets reales  
- 4.2.1 Dataset RELLIS-3D  
- 4.2.2 Dataset GOOSE  

### 4.3 Fusión LiDAR–visión  
- 4.3.1 Motivación  
- 4.3.2 Limitaciones sin alineamiento  

### 4.4 Alineamiento entre observaciones  
- 4.4.1 Planteamiento del problema  
- 4.4.2 Alineamiento en 2D (proyección LiDAR–imagen)  
- 4.4.3 Alineamiento en 3D (transformación de la nube)  
- 4.4.4 Estrategia de optimización (búsqueda de pose)  
- 4.4.5 Métricas de evaluación (IoU, soft IoU)  

---

## Capítulo 5. Experimentos y resultados

### 5.1 Configuración experimental  
- 5.1.1 Datos sintéticos  
- 5.1.2 Datos reales  

### 5.2 Resultados en datos sintéticos  

### 5.3 Resultados en datasets reales  

### 5.4 Análisis de resultados  
- 5.4.1 Evaluación cuantitativa  
- 5.4.2 Evaluación cualitativa  

---

## Capítulo 6. Conclusiones y trabajos futuros

### 6.1 Conclusiones  

### 6.2 Líneas futuras  

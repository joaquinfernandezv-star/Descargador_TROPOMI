# 🛰️ Descargador y Procesador Sentinel-5P Multigas

Esta aplicación es una herramienta integral escrita en Python para la descarga, procesamiento y análisis geoespacial de datos satelitales de Sentinel-5P y datos meteorológicos de ERA5.

A diferencia de versiones anteriores limitadas solo a NO₂, esta versión Multigas soporta múltiples contaminantes atmosféricos, aplicando criterios físicos diferenciados para su tratamiento.

## 📋 Características Principales

* **Multi-Gas:** Soporte para NO₂, CO, O₃, SO₂, CH₄ y HCHO.
* **Fusión de Datos:** Combina datos de columna satelital con la altura de la capa límite (BLH) de ERA5.
* **Criterio Científico:** Discrimina entre gases confinados a la capa límite y gases de troposfera gruesa.
* **Visualización:** Generación automática de mapas, histogramas y análisis de series temporales.
* **Batch Processing:** Capacidad de procesar múltiples regiones geográficas en una sola ejecución.

---

## 🔬 Fundamento Científico: Conversión Columna-Superficie

La herramienta permite convertir las densidades de columna (mol/m²) medidas por el satélite a concentraciones en superficie (ppb). Sin embargo, esta operación no es físicamente válida para todos los gases de la misma manera.

La aplicación utiliza la ecuación general:

$$C_{surf} = \frac{Columna_{sat}}{H_{mezcla}} \times Factores$$

Donde $H_{mezcla}$ suele ser la BLH (*Boundary Layer Height*). Basado en la literatura científica reciente (Petetin et al., Savanets et al.), la herramienta clasifica los gases en dos grupos:

### 🟢 Grupo 1: Gases Confinados a la Capa Límite (Transformación Recomendada)

Para estos gases, usar la BLH como altura de mezcla es una aproximación razonable de primer orden, ya que sus fuentes son superficiales y su vida media es corta.

| Gas | Nombre | Justificación |
| :--- | :--- | :--- |
| **NO₂** | Dióxido de Nitrógeno | La mayor parte de la masa troposférica se concentra en la PBL. El producto satelital está optimizado para la troposfera baja. |
| **HCHO** | Formaldehído | Vida media corta (horas). Su producción está ligada a COV en superficie. Distribución vertical decrece rápido con la altitud. |
| **SO₂** | Dióxido de Azufre | En contextos industriales/urbanos, se confina a la PBL. *Nota: No válido para plumas volcánicas inyectadas en la troposfera libre.* |

### 🔴 Grupo 2: Gases de "Troposfera Gruesa" (Transformación Restringida)

Para estos gases, dividir la columna total por la BLH sobreestima gravemente la concentración en superficie, ya que una gran parte de la masa del gas reside por encima de la capa límite.

| Gas | Nombre | Comportamiento |
| :--- | :--- | :--- |
| **CO** | Monóxido de Carbono | Vida larga (semanas). Se transporta a la troposfera media/superior. |
| **O₃** | Ozono | Perfil complejo con máximos en troposfera media y contribuciones estratosféricas. |
| **CH₄** | Metano | Distribución vertical homogénea pero columna dominada por la troposfera libre. |

> **⚠️ Nota de la UI:** Por defecto, la aplicación deshabilita la opción de transformación a superficie para el Grupo 2 (CO, O₃, CH₄) para evitar errores científicos, a menos que se introduzcan factores de corrección (*Shape Factors*) avanzados en el código.

---

## ⚙️ Instalación y Requisitos

### 0. Prerrequisitos

Se requiere Python 3.8+ instalado. Ejecuta el siguiente comando para instalar todas las dependencias geoespaciales y de interfaz gráfica:

```bash
python -m pip install tkcalendar matplotlib matplotlib-scalebar cmcrameri numpy pandas geopandas rasterio scipy contextily shapely sentinelhub pykrige cdsapi rioxarray netCDF4

```

### 1. Credenciales

El script requiere acceso a dos servicios de datos europeos. Debes configurar tus credenciales en el código o asegurarte de que los archivos de configuración existan en tu sistema:

1. **Copernicus Data Space (SentinelHub):**
* Regístrate en [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
* Configura `sh_client_id` y `sh_client_secret` en el script (`Descargador_Multigas_UI.py`).


2. **Climate Data Store (ERA5):**
* Regístrate en CDS.
* Crea un archivo `.cdsapirc` en tu carpeta de usuario (`C:\Users\{Usuario}\` o `~/.cdsapirc`) con tu URL y Key.



---

## 🚀 Guía de Uso

Ejecuta el script principal:

```bash
python Descargador_Multigas_UI.py

```

### Paso 1: Selección de Fecha 📅

Define el periodo temporal. El sistema soporta:

* **Mes/Año:** Promedios mensuales (recomendado para reducir ruido).
* **Día:** Eventos puntuales (sujeto a nubosidad).
* **Rango:** Promedios sobre periodos personalizados.

### Paso 2: Selección de Región 🌍

* **Lista Precargada:** Archivos `.geojson` ubicados en la carpeta `Regiones/`.
* **Manual:** Define un Bounding Box (Lat/Lon) directamente en la interfaz.

### Paso 3: Opciones de Procesamiento 🛠️

Aquí es donde seleccionas el gas y la metodología:

* **Gas a Procesar:** Selecciona NO₂, CO, O₃, SO₂, CH₄ o HCHO.
* **Transformación a Superficie:**
* *Disponible solo para Grupo 1 (NO₂, HCHO, SO₂).*
* **Método Petetin (Recomendado):** Descarga BLH horario/mensual de ERA5 y calcula pixel a pixel.
* **Método Savanets:** Asume una altura de mezcla constante (menos preciso).


* **Re-escalado (Kriging):** Interpola la imagen para suavizar pixelado y llenar huecos por nubes.
* **Formatos:** GeoTIFF (estándar), NetCDF4 (científico), ASCII (legacy).

### Paso 4: Ejecución y Visualización 📊

* **Nubes:** Usa el botón `☁️ Calc %` para verificar si la imagen es viable antes de descargar.
* **Iniciar Proceso:** Comienza la descarga y cálculo.
* **Resultados:** Se guardan en la carpeta `Resultados/` estructurada por Año > Región > Producto.

---

## 📂 Estructura de Salida

```text
Resultados/
├── Modelo/
│   └── 2024/
│       └── Santiago/
│           └── Datos_NO2_Enero/       # (GeoTIFFs crudos del satélite)
├── BLH/
│   └── 2024/
│       └── Santiago/
│           └── Datos_BLH_Enero/       # (Datos meteorológicos ERA5)
└── Calculos/
    └── 2024/
        └── Santiago/
            └── Concentracion_NO2/     # (Producto Final en ppb)
                ├── Mapa_Concentracion.png
                ├── Estadisticas.csv
                └── Raster_Final.tiff

```

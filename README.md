## 🚀 Uso de la Herramienta

Esta aplicación permite la descarga y procesamiento automatizado de datos satelitales (Sentinel-5P) y meteorológicos (ERA5) mediante una interfaz gráfica.

### 1\. Configuración Inicial

Antes de ejecutar, asegúrate de tener las credenciales de **Copernicus Data Space** y **CDS API (Climate Data Store)** configuradas en tu entorno o en el script.

  * Ejecuta el script principal:
    ```bash
    python Descargador_UI_NO2.py
    ```
  * *Nota:* El script requiere una carpeta llamada `Regiones/` en el mismo directorio, que contenga los archivos `.geojson` de las zonas de interés.

### 2\. Flujo de Trabajo

La interfaz se divide en 4 pasos secuenciales:

#### 1\. Selección de Fecha

Elige el rango temporal de análisis:

  * **Mes/Año:** Para análisis mensuales estándar.
  * **Año Completo:** Procesa los 12 meses de un año seleccionado.
  * **Día Puntual:** Para eventos específicos (mantiene la resolución nativa diaria).
  * **Rangos:** Permite definir periodos personalizados por días o meses.

#### 2\. Selección de Región

Define el área de interés (AOI):

  * **Lista Precargada:** Selecciona un polígono desde los archivos disponibles en la carpeta `Regiones/`.
  * **Manual (BBox):** Ingresa coordenadas manuales (Latitud/Longitud mínimas y máximas) y asigna un nombre a la zona.

#### 3\. Opciones de Procesamiento

Configura cómo se tratarán los datos:

  * **Transformación a Superficie:** Convierte la columna troposférica de $NO_2$ a concentración en superficie (ppb).
      * *Método H. Petetin:* Descarga dinámicamente la altura de la capa límite (BLH) de ERA5.
      * *Método Savanets:* Usa una altura constante de 10 km.
      * *Custom:* Permite ingresar un valor de altura fijo manual.
  * **Re-escalado (Kriging):** Interpola los píxeles para suavizar la imagen y cubrir huecos.
  * **Formatos de Salida:** Elige entre GeoTIFF, NetCDF4 o ASCII Grid.
  * **Compresión:** Opción para comprimir los datos crudos en `.zip` al finalizar.

#### 4\. Visualización y Ejecución

  * **Verificación de Nubosidad:** Usa el botón `☁️ Calcular % Nubes` para obtener una estimación rápida de la nubosidad en la zona antes de descargar.
  * **Botones de Acción:**
      * `Iniciar Proceso`: Ejecuta la configuración actual para la región seleccionada.
      * `Descargar todas las regiones`: Ejecuta el proceso en bucle para **todos** los archivos `.geojson` disponibles en la carpeta.

### 📂 Salida de Datos

Los resultados se guardan automáticamente en la carpeta `Resultados/`, organizados jerárquicamente por:

1.  **Año**
2.  **Nombre de la Región**
3.  **Tipo de Producto** (Modelo NO2, BLH, Cálculos de Concentración)

Cada ejecución genera mapas (`.png`), estadísticas (`.csv`) y los archivos ráster procesados.

## 🔗 Acceso a Resultados en la Nube
Puedes visualizar y descargar los ejemplos de resultados procesados en el siguiente enlace:
👉 [Ver Carpeta de Resultados en Google Drive](https://drive.google.com/drive/folders/1L0KB46bhG3BF9sIIjKRkHEh8tWrlT_Y4?usp=sharing)

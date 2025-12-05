# --- IMPORTACIONES ---
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import threading
import queue
import sys
import os
from pathlib import Path
import calendar
import json
import zipfile
from datetime import datetime # Necesario para el rango de meses y parseo de fechas

# Se necesita tkcalendar para los selectores de fecha
try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Librería Faltante", "La librería 'tkcalendar' no está instalada.\n\nPor favor, instálala ejecutando en tu terminal:\npip install tkcalendar")
    sys.exit(1)

# --- Importaciones de librerías geoespaciales y científicas ---
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import cmcrameri.cm as cmc
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    from rasterio.transform import from_origin
    from scipy.stats import norm
    # --- Importamos contextily para mapas base ---
    import contextily as cx
    # --- Importamos box de shapely para crear geometrías manuales ---
    from shapely.geometry import box
    # --------------------------------------------------
    from sentinelhub import (
        SHConfig, CRS, BBox, DataCollection, MimeType,
        SentinelHubRequest
    )
    from pykrige.ok import OrdinaryKriging
    import cdsapi
    import rioxarray
except ImportError as e:
    messagebox.showerror("Librería Faltante", f"Falta una librería necesaria: {e}.\n\nPor favor, instálala usando 'pip install <libreria>'.")
    sys.exit(1)

# ==============================================================================
# SECCIÓN 1: LÓGICA DE PROCESO Y CONFIGURACIÓN
# ==============================================================================

# --- RUTAS DINÁMICAS (NO MODIFICAR) ---
# Se detecta la ubicación del script para que sea portable
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback para entornos donde __file__ no está definido (como algunos IDEs interactivos)
    SCRIPT_DIR = Path.cwd()

BASE_OUTPUT_PATH = SCRIPT_DIR / "Resultados"
BASE_GEOJSON_PATH = SCRIPT_DIR / "Regiones"

# Se crea la carpeta de resultados si no existe
BASE_OUTPUT_PATH.mkdir(exist_ok=True)
# Aseguramos que la carpeta de regiones exista para guardar manuales
BASE_GEOJSON_PATH.mkdir(exist_ok=True)


# --- CONFIGURACIÓN DE SENTINEL HUB ---
try:
    config = SHConfig()
    if not config.sh_client_id or not config.sh_client_secret:
        print("Configurando credenciales de Copernicus Data Space...")
        config.sh_client_id = "sh-2279fd56-dabb-4e4d-ae5b-b71ce5fc5c09"
        config.sh_client_secret = "9c94Zs5JMkwIkwGqyBJGCSXigh9jslVP"
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.save("cdse")
    print("Configuración de SentinelHub cargada.")
except Exception as e:
    print(f"Error configurando SentinelHub: {e}")

# --- DEFINICIONES Y CONSTANTES ---
data_5p = DataCollection.SENTINEL5P.define_from("5p", service_url=config.sh_base_url)

# --- Evalscript SIMPLE (Original) ---
evalscript_raw = """
//VERSION=3
function setup() { return { input: ["NO2"], output: { bands: 1, sampleType: "FLOAT32" }, mosaicking: "SIMPLE" }; }
function evaluatePixel(samples) { return [samples.NO2]; }
"""

# --- Evalscript PROMEDIO (Nuevo - Mosaicking ORBIT) ---
evalscript_mean_mosaic = """
//VERSION=3
function setup() {
    return {
        input: ["NO2", "dataMask"],
        output: {
            bands: 1,
            sampleType: "FLOAT32",
        },
        mosaicking: "ORBIT"
    };
}

function isClear(sample) {
    return sample.dataMask == 1;
}

function sum(array) {
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
        sum += array[i].NO2;
    }
    return sum;
}

function evaluatePixel(samples) {
    const clearTs = samples.filter(isClear)
    if (clearTs.length == 0) return [NaN];
    const mean = sum(clearTs) / clearTs.length
    return [mean]
}
"""

# --- Evalscript para NUBOSIDAD ---
evalscript_cloud = """
//VERSION=3
function setup() {
    return {
        input: ["CLOUD_FRACTION"],
        output: { bands: 1, sampleType: "FLOAT32" },
        mosaicking: "SIMPLE"
    };
}
function evaluatePixel(samples) {
    return [samples.CLOUD_FRACTION];
}
"""

meses_dict = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}
meses_es_lower = {k: v.lower() for k, v in meses_dict.items()}

try:
    import cmcrameri.cm as cmc; tiene_cmcrameri = True
except ImportError:
    tiene_cmcrameri = False
    
paletas_colores = {
    'cividis': 'Azul a amarillo (gradiente)', 'viridis': 'Púrpura, azul, verde, amarillo (gradiente)',
    'turbo': 'Arcoiris (gradiente)', 'inferno': 'Negro, rojo, amarillo (gradiente)',
    'batlow' if tiene_cmcrameri else 'plasma': 'Perceptual (batlow) o alternativa (plasma)'
}

# --- FUNCIONES AUXILIARES ---
def get_available_regions():
    """Escanea la carpeta de GeoJSONs y devuelve una lista de nombres de regiones."""
    if not BASE_GEOJSON_PATH.is_dir():
        return []
    return sorted([f.stem for f in BASE_GEOJSON_PATH.glob("*.geojson")])

def geojson_to_coords(geojson_path: str):
    try:
        gdf = gpd.read_file(geojson_path)
        bounds = gdf.total_bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]] # minx, miny, maxx, maxy
    except Exception as e:
        print(f"Error leyendo el archivo GeoJSON {geojson_path}: {e}")
        return None

def comprimir_directorio(directorio_origen, archivo_destino):
    with zipfile.ZipFile(archivo_destino, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directorio_origen):
            for file in files:
                archivo_completo = os.path.join(root, file)
                ruta_relativa = os.path.relpath(archivo_completo, os.path.dirname(directorio_origen))
                zipf.write(archivo_completo, ruta_relativa)

# --- FUNCIÓN: CÁLCULO DE NUBOSIDAD ---
def calcular_estadisticas_nubosidad(time_start, time_end, route):
    """
    Descarga una imagen rápida de CLOUD_FRACTION y calcula el promedio.
    Retorna el porcentaje promedio (0-100).
    """
    print(f"☁️  Analizando nubosidad para {time_start} - {time_end}...")
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None

    # Usamos una resolución más baja para que el cálculo sea rápido (ej: 2000x1500 px aprox)
    request_cloud = SentinelHubRequest(
        evalscript=evalscript_cloud,
        input_data=[SentinelHubRequest.input_data(
            data_collection=data_5p,
            time_interval=(time_start, time_end),
            other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '50', 'timeliness': 'OFFL'}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)),
        resolution=(2000, 2000), # Resolución media para velocidad
        config=config
    )
    
    try:
        data = request_cloud.get_data()[0] # Obtener array numpy directamente
        
        # data tiene shape (height, width, bands). Band is index 0.
        # Los valores vienen de 0.0 a 1.0 (fracción)
        cloud_values = data.flatten()
        
        # Filtramos valores nan o nodata si los hubiera
        valid_clouds = cloud_values[~np.isnan(cloud_values)]
        
        if valid_clouds.size == 0:
            return 0.0
            
        mean_fraction = np.mean(valid_clouds)
        mean_percent = mean_fraction * 100
        
        print(f"✅ Nubosidad calculada: {mean_percent:.2f}%")
        return mean_percent
        
    except Exception as e:
        print(f"❌ Error calculando nubosidad: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---
def datos_mes_no2(time_start, time_end, route, output_name, evalscript_override=None):
    """
    Descarga datos de NO2.
    evalscript_override: Permite especificar si se usa el script 'Simple' o 'Mean Mosaic'.
    """
    area = Path(route).stem
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None
    
    year_folder = time_start[:4]
    data_folder = BASE_OUTPUT_PATH / f"Modelo/{year_folder}/{area}/{output_name}"
    data_folder.mkdir(parents=True, exist_ok=True)
    
    # Seleccionar evalscript (Por defecto RAW/Simple si no se especifica)
    script_to_use = evalscript_override if evalscript_override else evalscript_raw
    
    request_raw = SentinelHubRequest(
        evalscript=script_to_use, 
        input_data=[SentinelHubRequest.input_data(
            data_collection=data_5p, 
            time_interval=(time_start, time_end), 
            other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '75', 'timeliness': 'OFFL'}}
        )], 
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)], 
        bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)), 
        resolution=(5500, 3500), 
        config=config, 
        data_folder=str(data_folder)
    )
    request_raw.get_data(save_data=True)
    return request_raw

def descargar_blh_era5(year, month, area_coords, output_path):
    """Descarga el promedio mensual de BLH."""
    print("🛰️  Conectando al Copernicus Climate Data Store (CDS) para descargar BLH (Promedio Mensual)...")
    month_str = f"{month:02d}"
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {'product_type': 'monthly_averaged_reanalysis', 'variable': 'boundary_layer_height', 
             'year': str(year), 
             'month': month_str,
             'time': '00:00', 'area': area_coords, 'format': 'netcdf'},
            output_path)
        print(f"✅ Descarga BLH Mensual completada: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error durante la descarga de BLH mensual desde CDS: {e}")
        return False

# --- NUEVA FUNCIÓN INTEGRADA ---
def descargar_blh_era5_diario(start_date, end_date, area_coords, output_path):
    """Descarga Horaria (13:00) para un día específico."""
    # Nota: start_date se espera que sea un objeto datetime
    print(f"📡 Solicitando ERA5 Horario (13:00) para {start_date}...")
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'boundary_layer_height',
                'year': str(start_date.year),
                'month': f"{start_date.month:02d}",
                'day': f"{start_date.day:02d}",
                'time': '13:00',
                'area': area_coords,
                'format': 'netcdf',
            },
            output_path
        )
        print("✅ Descarga BLH Diario completada.")
        return True
    except Exception as e:
        print(f"❌ Error ERA5 Diario: {e}")
        return False
# -------------------------------

def convertir_nc_a_tiff(netcdf_path, tiff_path):
    try:
        data_array = rioxarray.open_rasterio(netcdf_path, variable='blh').squeeze()
        data_array.rio.write_crs("EPSG:4326", inplace=True)
        data_array.rio.to_raster(tiff_path, driver='GTiff')
        print(f"🔄 Archivo NetCDF (BLH) convertido a GeoTIFF: {tiff_path}")
        return True
    except Exception as e:
        print(f"❌ Error al convertir NetCDF (BLH) a TIFF: {e}")
        return False

def tag_nc_with_crs(netcdf_path, output_nc_path):
    """
    Abre un NetCDF de BLH, le asigna CRS EPSG:4326 y lo guarda de nuevo como NetCDF.
    """
    try:
        data_array = rioxarray.open_rasterio(netcdf_path, variable='blh').squeeze()
        data_array.rio.write_crs("EPSG:4326", inplace=True)
        data_array.rio.to_raster(output_nc_path, driver='NETCDF')
        print(f"🏷️  Archivo BLH NetCDF etiquetado con CRS: {output_nc_path}")
        return True
    except Exception as e:
        print(f"❌ Error al etiquetar NetCDF (BLH) con CRS: {e}")
        return False

def procesar_concentracion_no2(ruta_no2_crudo, ruta_blh_crudo, output_dir, region_nombre, año, mes_nombre_es, formato_salida="GeoTIFF", metodo_transform="petetin", valor_custom=None, suffix=""):
    """
    Calcula la concentración de NO2 en superficie.
    metodo_transform: "petetin" (usa archivo BLH), "savanets" (h=10000m), "custom" (h=valor_custom)
    suffix: Sufijo para diferenciar archivos (ej: "_regrid")
    """
    print(f"\n---\n🔬 Iniciando cálculo de concentración de NO₂ para: {mes_nombre_es} {año} (Modo: {metodo_transform})")
    try:
        with rasterio.open(ruta_no2_crudo) as src_no2:
            profile, ccol_array = src_no2.profile, src_no2.read(1).astype(np.float32)
            if src_no2.nodata is not None: ccol_array[ccol_array == src_no2.nodata] = np.nan
            
            h_resampled = None

            # --- LÓGICA DE SELECCIÓN DE H (DENOMINADOR) ---
            if metodo_transform == "petetin":
                # Modo H. Petetin: Usa el archivo ráster de BLH
                if not ruta_blh_crudo:
                    print("❌ Error: Se seleccionó modo Petetin pero no se proporcionó archivo BLH.")
                    return None
                
                print(f"    Usando BLH dinámico desde: {Path(ruta_blh_crudo).name}")
                with rasterio.open(ruta_blh_crudo) as src_blh:
                    h_resampled = np.empty_like(ccol_array)
                    print("    Alineando grilla BLH con NO₂ (reproyección)...")
                    reproject(
                        source=rasterio.band(src_blh, 1),
                        destination=h_resampled,
                        src_transform=src_blh.transform,
                        src_crs=src_blh.crs,
                        dst_transform=src_no2.transform,
                        dst_crs=src_no2.crs,
                        resampling=Resampling.bilinear
                    )
                    h_resampled[h_resampled <= 0] = np.nan

            elif metodo_transform == "savanets":
                # Modo Savanets: Altura constante 10km (10000m)
                print("    Usando constante Savanets: 10,000 metros.")
                h_resampled = 10000.0
            
            elif metodo_transform == "custom":
                # Modo Custom: Valor ingresado por usuario
                if valor_custom is None:
                    print("❌ Error: Modo Custom seleccionado pero sin valor definido.")
                    return None
                print(f"    Usando valor personalizado: {valor_custom} metros.")
                h_resampled = float(valor_custom)
            
            else:
                print(f"❌ Error: Método de transformación desconocido: {metodo_transform}")
                return None
            
            # --- CÁLCULO DE CONCENTRACIÓN ---
            M, A, PPB_FACTOR = 46.01, 1e6, 1.88
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Si h_resampled es escalar (Savanets/Custom), numpy broadcast es automático
                concentracion_ug_m3 = (ccol_array / h_resampled) * M * A
                concentracion_ppb = concentracion_ug_m3 / PPB_FACTOR
                
            if profile.get('nodata') is not None:
                concentracion_ppb[np.isnan(concentracion_ppb)] = profile['nodata']

            # --- INICIO BLOQUE DE FORMATOS DE SALIDA ---
            # Construimos el nombre base incluyendo el sufijo (ej: ..._enero_regrid)
            nombre_archivo_base = f"Concentracion_ppb_NO2_{region_nombre}_{año}_{mes_nombre_es}{suffix}"

            if formato_salida == "NetCDF4":
                template_raster = rioxarray.open_rasterio(ruta_no2_crudo)
                data_con_banda = concentracion_ppb.astype(np.float32)[np.newaxis, :, :]
                data_array = template_raster.copy(data=data_con_banda)
                data_array = data_array.rename('concentracion_no2_ppb')
                
                if profile.get('nodata') is not None:
                    data_array = data_array.rio.write_nodata(profile['nodata'])
                
                output_path = output_dir / f"{nombre_archivo_base}.nc"
                data_array.rio.to_raster(output_path, driver="NETCDF")

            elif formato_salida == "ASCII Grid (.asc)":
                output_path = output_dir / f"{nombre_archivo_base}.asc"
                
                # Definir un valor NODATA estándar para ASCII Grid
                nodata_value = -9999.0
                
                # Reemplazar NaNs en los datos con este valor
                concentracion_para_asc = np.nan_to_num(concentracion_ppb, nan=nodata_value)
                
                # Actualizar el perfil para el driver AAIGrid
                profile.update(
                    driver="AAIGrid",
                    dtype=rasterio.float32,
                    nodata=nodata_value
                )
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(concentracion_para_asc.astype(rasterio.float32), 1)

            else: # Default to GeoTIFF
                output_path = output_dir / f"{nombre_archivo_base}.tiff"
                profile.update(dtype=rasterio.float32)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(concentracion_ppb.astype(rasterio.float32), 1)
            # --- FIN BLOQUE DE FORMATOS DE SALIDA ---

            print(f"✅ Cálculo de concentración completado: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"❌ Error al procesar la concentración de NO₂: {e}")
        return None

# --- FUNCIÓN (Regrid) ---
def regrid_geotiff(input_tiff_path, grid_resolution=100):
    """
    Rejilla (re-grid) un GeoTIFF usando interpolación Kriging.
    """
    carpeta = os.path.dirname(input_tiff_path)
    output_tiff_path = os.path.join(carpeta, "response_regrid.tiff")

    try:
        print(f"Cargando GeoTIFF: {input_tiff_path}")
        with rasterio.open(input_tiff_path) as ds:
            band1 = ds.read(1)
            transform = ds.transform
            crs = ds.crs
            nodata = ds.nodata

            if nodata is not None:
                if not np.issubdtype(band1.dtype, np.floating):
                    band1 = band1.astype(np.float32)
                band1[band1 == nodata] = np.nan
            else:
                if not np.issubdtype(band1.dtype, np.floating):
                    band1 = band1.astype(np.float32)

            coords, vals = [], []
            print("Extrayendo coordenadas y valores válidos...")
            filas, columnas = band1.shape
            for fila in range(filas):
                for col in range(columnas):
                    val = band1[fila, col]
                    if not np.isnan(val):
                        lon, lat = transform * (col + 0.5, fila + 0.5)
                        coords.append((lon, lat))
                        vals.append(val)

        if not vals:
            print("No se encontraron datos válidos. Interpolación cancelada.")
            return None

        df = pd.DataFrame(coords, columns=["lon", "lat"])
        df["value"] = vals
        print(f"Se hallaron {len(df)} puntos para interpolar.")

        lon_grid = np.linspace(df["lon"].min(), df["lon"].max(), grid_resolution)
        lat_grid = np.linspace(df["lat"].max(), df["lat"].min(), grid_resolution)
        print(f"Creando malla {grid_resolution}x{grid_resolution}...")

        ok = OrdinaryKriging(
            df["lon"].values,
            df["lat"].values,
            df["value"].values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False
        )
        print("Ejecutando Kriging...")
        interpolado, _ = ok.execute("grid", lon_grid, lat_grid)
        print("Interpolación terminada.")

        if grid_resolution > 1:
             res_x = (lon_grid.max() - lon_grid.min()) / (len(lon_grid) - 1)
             res_y = (lat_grid.min() - lat_grid.max()) / (len(lat_grid) - 1) # Y_min - Y_max
        else:
             res_x = res_y = 0

        left_edge = lon_grid.min() - res_x / 2
        top_edge  = lat_grid.max() - res_y / 2 
        
        nuevo_transform = from_origin(left_edge, top_edge, res_x, abs(res_y))

        print(f"Guardando GeoTIFF interpolado en: {output_tiff_path}")
        with rasterio.open(
            output_tiff_path,
            "w",
            driver="GTiff",
            height=interpolado.shape[0],
            width=interpolado.shape[1],
            count=1,
            dtype=interpolado.dtype,
            crs=crs,
            transform=nuevo_transform
        ) as dst:
            dst.write(interpolado, 1)

        print("¡GeoTIFF guardado con éxito!")
        return output_tiff_path

    except FileNotFoundError:
        print("Error: archivo de entrada no encontrado.")
        return None
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None
# --- FIN DE LA NUEVA FUNCIÓN ---

# --- FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS ---
def generar_mapa_con_leyenda(ruta_tiff, ruta_geojson, title_date, year, cmap='viridis', alpha=0.75, producto="NO₂", unidad="(mol/m²)", return_fig=False):
    try:
        gdf = gpd.read_file(ruta_geojson).dissolve()
        region_nombre = Path(ruta_geojson).stem.replace("_", " ").title()
        with rasterio.open(ruta_tiff) as src:
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            try:
                data, out_transform = mask(dataset=src, shapes=gdf.geometry, crop=True, nodata=np.nan)
            except ValueError: print(f"❌ El polígono de {region_nombre} no se superpone con el ráster de {producto}."); return None
            data = data[0]
            if np.all(np.isnan(data)): print(f"⚠️ Advertencia: No hay datos válidos para {producto} en la región seleccionada."); return None
            
            bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], out_transform)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]] # (left, right, bottom, top)

        output_path = Path(ruta_tiff).with_suffix('.png')
        
        if 'regrid' in Path(ruta_tiff).stem.lower():
            procesamiento = "Re-escalado (Kriging)"
        elif 'concentracion' in Path(ruta_tiff).stem.lower():
            procesamiento = "Calculado"
        else:
            procesamiento = "Crudo"

        fig, ax = plt.subplots(figsize=(12, 10))
        cmap_obj = cmc.batlow if tiene_cmcrameri and cmap == 'batlow' else plt.get_cmap(cmap)
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        
        # Ploteo de datos (recortados según la máscara)
        img = ax.imshow(data, extent=extent, cmap=cmap_obj, norm=norm, alpha=alpha, zorder=10)
        
        # Ploteo del borde
        gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, linestyle='--', zorder=11)
        
        # --- INTEGRACIÓN CONTEXTILY (Mapa Base) ---
        try:
            # Añadir mapa base (OpenStreetMap). 
            # crs=gdf.crs asegura que el mapa se reproyecte al sistema del gráfico actual.
            # source=cx.providers.OpenStreetMap.Mapnik es el estilo estándar de OSM.
            print("    Añadiendo mapa base OpenStreetMap...")
            cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"⚠️  No se pudo cargar el mapa base de Contextily: {e}")
        # ------------------------------------------

        ax.set_title(f"{producto} en {region_nombre} • {title_date.capitalize()} {year} • Datos {procesamiento}", fontsize=16, pad=15)
        ax.set_xlabel("Longitud", fontsize=14); ax.set_ylabel("Latitud", fontsize=14); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.2)
        cbar = plt.colorbar(img, cax=cax); cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=20, fontsize=14)
        plt.tight_layout(); plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"🖼️ Mapa de {producto} guardado en: {output_path}")
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception as e: print(f"❌ Error al generar el mapa para {ruta_tiff}: {e}"); return None

# ==============================================================================
# ESTA ES LA FUNCIÓN CORREGIDA
# ==============================================================================
def generar_mapa_comparativo(file_paths, aoi_path, producto, unidad, cmap, title_suffix, return_fig=False):
    """
    Genera un mapa con múltiples subplots (uno para cada archivo en file_paths),
    compartiendo una única escala de color y ejes.
    """
    print(f"\n🗺️  Generando mapa comparativo para {producto}...")
    if not file_paths:
        print("     No se encontraron archivos para el mapa comparativo.")
        return None
    
    try:
        gdf = gpd.read_file(aoi_path).dissolve()
        region_nombre = Path(aoi_path).stem.replace("_", " ").title()
        
        print("     Calculando escala de color global...")
        vmin, vmax = np.inf, -np.inf
        
        all_data = {} 
        valid_extents = []
        target_crs = None 
        
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                if target_crs is None:
                    target_crs = src.crs
                    
                if gdf.crs != src.crs: gdf_proj = gdf.to_crs(src.crs)
                else: gdf_proj = gdf
                
                try:
                    data, out_transform = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    data = data[0]
                    if np.all(np.isnan(data)):
                        all_data[file_path] = {'data': None, 'extent': None}
                        continue 
                    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], out_transform)
                    extent = [bounds[0], bounds[2], bounds[1], bounds[3]] # (left, right, bottom, top)
                    all_data[file_path] = {'data': data, 'extent': extent}
                    
                    valid_extents.append(extent)
                    
                    vmin = min(vmin, np.nanmin(data))
                    vmax = max(vmax, np.nanmax(data))
                except ValueError:
                    print(f"     Advertencia: El polígono no se superpone con {file_path.name}, se omite.")
                    all_data[file_path] = {'data': None, 'extent': None}

        if not valid_extents: 
            print("❌ Error: No se encontraron datos válidos en ningún archivo para el mapa comparativo.")
            return None
            
        print(f"     Escala global (Min/Max): {vmin:.3e} / {vmax:.3e}")
        
        global_left = min(e[0] for e in valid_extents)
        global_right = max(e[1] for e in valid_extents)
        global_bottom = min(e[2] for e in valid_extents)
        global_top = max(e[3] for e in valid_extents)
        print(f"     BBOX Global: [{global_left:.2f}, {global_bottom:.2f}, {global_right:.2f}, {global_top:.2f}]")
        
        if target_crs and gdf.crs != target_crs:
            print(f"     Reproyectando GeoJSON a {target_crs} para el ploteo...")
            gdf_plot = gdf.to_crs(target_crs)
        else:
            gdf_plot = gdf
            
        n = len(file_paths)
        if n <= 4:
            nrows, ncols = 1, n
        elif n <= 8:
            nrows, ncols = 2, (n + 1) // 2
        elif n <= 12:
            nrows, ncols = 3, 4
        else: 
            nrows, ncols = (n + 3) // 4, 4 
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 7)) 
        axs = np.atleast_1d(axs).flatten() 
        
        cmap_obj = cmc.batlow if tiene_cmcrameri and cmap == 'batlow' else plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        img = None 
        
        print("     Ploteando mapas individuales...")
        for i, file_path in enumerate(file_paths):
            ax = axs[i]
            plot_data = all_data[file_path]
            
            # --- INICIO LÓGICA DE TÍTULO (CORREGIDA) ---
            title = next((m for m in meses_dict.values() if m.lower() in str(file_path).lower()), file_path.stem) # Plan B: el nombre del archivo (ej: "response")
            try:
                # Plan A: el nombre de la CARPETA (ej: "01_Datos_NO2_enero_2019")
                folder_name = file_path.parent.name
                parts = folder_name.split('_') 
                if len(parts) >= 2:
                    mes_nombre_raw = parts[-2]
                    ano_raw = parts[-1]
                    
                    # Validación: Asegurarse que 'ano_raw' sea un año
                    if ano_raw.isdigit() and len(ano_raw) == 4:
                        title = f"{mes_nombre_raw.capitalize()} {ano_raw}"
            except Exception as e:
                print(f"Advertencia: No se pudo extraer el título de '{folder_name}'. Usando stem. Error: {e}")
            # --- FIN LÓGICA DE TÍTULO ---
            
            if plot_data['data'] is not None:
                # Ploteamos datos recortados (zorder más alto para estar encima del mapa)
                img = ax.imshow(plot_data['data'], extent=plot_data['extent'], cmap=cmap_obj, norm=norm, alpha=0.8, zorder=10)
                gdf_plot.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, linestyle='--', zorder=11)
                ax.set_title(title, fontsize=11) 
                
                # --- INTEGRACIÓN CONTEXTILY (Mapa Comparativo) ---
                try:
                    cx.add_basemap(ax, crs=gdf_plot.crs, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
                except Exception as e:
                    pass # Fallo silencioso en comparativo para no saturar consola
                # -------------------------------------------------

            else:
                ax.set_title(f"{title}\n(Sin datos)", fontsize=12)
                ax.set_facecolor('0.95') 
            
            ax.set_xlim(global_left, global_right)
            ax.set_ylim(global_bottom, global_top)
            
            ax.tick_params(labelbottom=False, labelleft=False)

            if i // ncols == nrows - 1:
                ax.tick_params(labelbottom=True)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

            # --- INICIO CORRECCIÓN DE EJE Y ---
            # Solo mostrar las etiquetas Y en la columna izquierda
            if i % ncols == 0:
                ax.tick_params(labelleft=True)
                plt.setp(ax.get_yticklabels(), fontsize=10)
            # --- FIN CORRECCIÓN DE EJE Y ---

        for i in range(n, len(axs)):
            axs[i].set_visible(False)
            
        fig.suptitle(f"{producto} en {region_nombre} • {title_suffix}", fontsize=18) 
        
        if img: 
            
            # --- INICIO CORRECCIÓN DE MÁRGENES ---
            # Dejamos espacio a la izquierda (0.04) y abajo (0.05) para las etiquetas
            fig.tight_layout(rect=[0.04, 0.05, 0.85, 0.95])
            fig.canvas.draw() 

            # Posición del Colorbar (sin cambios)
            top_axes = axs[:ncols]
            global_top = max(ax.get_position().y1 for ax in top_axes if ax.get_visible())
            last_row_start_index = (nrows - 1) * ncols
            bottom_axes = axs[last_row_start_index : n] 
            global_bottom = min(ax.get_position().y0 for ax in bottom_axes if ax.get_visible())
            cbar_height = (global_top - global_bottom*2.3) 
            cbar_ax = fig.add_axes([0.87, global_bottom + 0.1025 , 0.03, cbar_height])
            cbar = fig.colorbar(img, cax=cbar_ax)
            cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=25, fontsize=14)
            
            # --- INICIO CORRECCIÓN DE ETIQUETAS "LONGITUD" Y "LATITUD" ---
            # Usamos etiquetas "superiores" (centrales) para toda la figura
            # Esto reemplaza a los ax.set_xlabel y ax.set_ylabel individuales
            fig.supxlabel('Longitud', fontsize=16, y=0.02) # y=0.02 lo pone cerca del borde inferior
            fig.supylabel('Latitud', fontsize=16, x=0.01) # x=0.01 lo pone cerca del borde izquierdo
            # --- FIN CORRECCIÓN DE ETIQUETAS ---

        else:
            fig.tight_layout(rect=[0.04, 0.05, 1.0, 0.95])
        
        # --- FIN DE LAS CORRECCIONES ---

        output_path = file_paths[0].parent.parent / f"Mapa_Comparativo_{producto}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"🖼️  Mapa comparativo guardado en: {output_path}")
        
        if return_fig: return fig
        else: plt.close(fig); return None

    except Exception as e:
        print(f"❌ Error al generar el mapa comparativo: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_tiff_statistics(tiff_path: str, return_fig=False):
    try:
        with rasterio.open(tiff_path) as src:
            image_data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None: image_data[image_data == nodata] = np.nan
            pixel_values = image_data[~np.isnan(image_data)]
        
        if pixel_values.size == 0: 
            print(f"No se encontraron datos válidos en {tiff_path}.") 
            return

        # Calcular estadísticas básicas
        mean_val, std_val = np.nanmean(pixel_values), np.nanstd(pixel_values)
        median_val = np.nanmedian(pixel_values)  # <--- NUEVO: Cálculo de mediana

        # Crear DataFrame con la Mediana incluida
        stats_df = pd.DataFrame({
            'Statistic': ['Minimum', 'Maximum', 'Median', 'Mean', 'Std Dev', 'Data Points'],
            'Value': [
                np.nanmin(pixel_values), 
                np.nanmax(pixel_values), 
                f"{median_val:.3e}",  # <--- NUEVO: Mediana formateada
                f"{mean_val:.3e}", 
                f"{std_val:.3e}", 
                pixel_values.size
            ]
        })
        
        csv_output_path = f"{Path(tiff_path).with_suffix('')}_statistics.csv"
        stats_df.to_csv(csv_output_path, index=False)
        print(f"📊 Estadísticas guardadas en: {csv_output_path}")

        # Generación del gráfico (sin cambios mayores)
        fig = plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=50, density=True, alpha=0.7, color='skyblue', label='Distribución de Datos')
        x_norm = np.linspace(pixel_values.min(), pixel_values.max(), 100)
        pdf_fitted = norm.pdf(x_norm, mean_val, std_val)
        plt.plot(x_norm, pdf_fitted, 'r-', linewidth=2, label=f'Distribución Normal (μ={mean_val:.2e}, σ={std_val:.2e})')
        plt.title(f"Distribución de Valores para {Path(tiff_path).stem}")
        plt.xlabel("Valor del Píxel"); plt.ylabel("Densidad"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_output_path = f"{Path(tiff_path).with_suffix('')}_distribution.png"
        plt.savefig(plot_output_path)
        print(f"📈 Gráfico de distribución guardado en: {plot_output_path}")
        
        if return_fig: return fig
        else: plt.close(fig); return None

    except Exception as e: 
        print(f"Ocurrió un error analizando {tiff_path}: {e}") 
        return None

# --- FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---
def run_processing(params, cancel_event):
    matplotlib.use('Agg')
    
    aoi_path = params['aoi_path']
    region_nombre = Path(aoi_path).stem
    time_start, time_end = params['start_date'], params['end_date']
    
    ruta_crudo_no2 = None
    ruta_blh_para_calculo = None 
    ruta_no2_para_procesar = None
    ruta_blh_para_analisis = None
    ruta_concentracion_final = None

    print(f"\n\n--- 🗓️  PROCESANDO {region_nombre.replace('_', ' ').title()} para el período de {time_start} a {time_end} ---")

    # --- 1. PROCESO PARA NO₂ (SENTINEL-5P) ---
    print(f"\n📥 Descargando datos de NO₂ (C_col) para el período...")
    
    # --- SELECCIÓN DE EVALSCRIPT ---
    # Si estamos en modos mensuales/anuales/rango, usamos el script de PROMEDIOS (Mean Mosaic)
    # Si es "dia" (puntual), mantenemos el script RAW (Simple Mosaic)
    use_mean_script = params['choice_mode'] in ['mes', 'anio', 'rango_meses', 'rango']
    script_to_use = evalscript_mean_mosaic if use_mean_script else evalscript_raw
    if use_mean_script:
        print("ℹ️  Usando Evalscript de Promedios (Mosaicking ORBIT).")
    else:
        print("ℹ️  Usando Evalscript Simple (Default).")

    request_no2 = datos_mes_no2(time_start, time_end, aoi_path, params['output_name'], evalscript_override=script_to_use)
    
    if request_no2 and request_no2.get_filename_list():
        carpeta_base_no2 = Path(request_no2.data_folder).resolve()
        ruta_crudo_no2 = (carpeta_base_no2 / request_no2.get_filename_list()[0]).resolve()
        print(f"🗂️  Archivo base NO₂ (crudo): {ruta_crudo_no2}")
        
        ruta_no2_para_procesar = ruta_crudo_no2 

        if params['do_regrid']:
            print(f"\n🔄 Iniciando re-escalado (Kriging) para: {ruta_crudo_no2}...")
            ruta_regrid_no2_str = regrid_geotiff(str(ruta_crudo_no2)) 
            
            if ruta_regrid_no2_str:
                print(f"✅ Re-escalado completado: {ruta_regrid_no2_str}")
                ruta_no2_para_procesar = Path(ruta_regrid_no2_str).resolve()
            else:
                print(f"⚠️  Advertencia: El re-escalado falló. Se continuará con el archivo crudo.")
        
        if not params.get('do_comparative_map', False):
            if params['estadisticas']:
                if fig := analyze_tiff_statistics(str(ruta_no2_para_procesar), return_fig=params['show_plots']):
                    params['fig_queue'].put(fig)
            if params['generar_mapas']:
                if fig := generar_mapa_con_leyenda(ruta_no2_para_procesar, aoi_path, params['title_date'], params['year'], cmap=params['palette'], producto="NO₂ (Columna Troposférica)", unidad="(mol/m²)", return_fig=params['show_plots']):
                    params['fig_queue'].put(fig)
    else:
        print(f"⚠️  Advertencia: No se obtuvieron datos de NO₂ para el período seleccionado.")

    # --- CHEQUEO DE CANCELACIÓN ---
    if cancel_event.is_set(): print("--- 🛑 Cancelado después del paso NO₂. ---"); return None, None, None

    # --- 2. PROCESO PARA BLH (ERA5) --- 
    # Solo descargar si el método elegido es Petetin (que requiere BLH)
    if params.get('transform_method') == 'petetin':
        print(f"\n🌍 Descargando datos de BLH (H) para: {params['title_date']} {params['year']}")
        bounding_box = geojson_to_coords(aoi_path)
        if not bounding_box:
            print("❌ No se pudieron obtener las coordenadas del GeoJSON. Se omite la descarga de BLH.")
        else:
            min_x, min_y, max_x, max_y = bounding_box
            era5_area_coords = [max_y, min_x, min_y, max_x]
            
            blh_output_dir = BASE_OUTPUT_PATH / f"BLH/{params['year']}/{region_nombre}/{params['output_name_blh']}"
            blh_output_dir.mkdir(parents=True, exist_ok=True)
            nombre_base_blh = f"BLH_ERA5_{region_nombre}_{params['year']}_{params['title_date']}"
            netcdf_file = blh_output_dir / f"{nombre_base_blh}.nc"
            tiff_file_blh = blh_output_dir / f"{nombre_base_blh}.tiff"

            # --- LOGICA DE DISCRIMINACION DE DESCARGA (MENSUAL vs DIARIA) ---
            exito_descarga = False
            
            if params.get('choice_mode') == 'dia':
                 # Modo Diario
                 try:
                     # Parsear fecha string a datetime para la función
                     fecha_dt = datetime.strptime(params['start_date'], '%Y-%m-%d')
                     exito_descarga = descargar_blh_era5_diario(fecha_dt, fecha_dt, era5_area_coords, netcdf_file)
                 except Exception as e:
                     print(f"❌ Error preparando fecha para descarga diaria: {e}")
                     exito_descarga = False
            else:
                 # Modo Mensual (Default)
                 if 'month' in params:
                     exito_descarga = descargar_blh_era5(params['year'], params['month'], era5_area_coords, netcdf_file)
                 else:
                     print("⚠️ Modo no compatible con descarga BLH automática (se requiere mes específico o día puntual).")

            if exito_descarga:
                netcdf_file_tagged = blh_output_dir / f"{nombre_base_blh}_tagged.nc"
                ruta_blh_para_analisis = None 
                
                if params['formato_salida'] == "GeoTIFF":
                    print(f"🔄 Convirtiendo BLH NetCDF a GeoTIFF (para cálculo)...")
                    if convertir_nc_a_tiff(netcdf_file, tiff_file_blh):
                        ruta_blh_para_calculo = tiff_file_blh 
                        ruta_blh_para_analisis = tiff_file_blh
                    else:
                        print("❌ Error en conversión de BLH, se omite.")
                        ruta_blh_para_calculo = None
                else: # Formato de salida es NetCDF4 o ASCII Grid. Usaremos NetCDF como intermedio.
                    print(f"ℹ️  Etiquetando BLH NetCDF con CRS (para cálculo)...")
                    if tag_nc_with_crs(netcdf_file, netcdf_file_tagged):
                        ruta_blh_para_calculo = netcdf_file_tagged
                        ruta_blh_para_analisis = netcdf_file_tagged
                    else:
                        print("❌ Error al etiquetar BLH con CRS, se omite.")
                        ruta_blh_para_calculo = None
                
                if ruta_blh_para_analisis:
                    if not params.get('do_comparative_map', False):
                        if params['estadisticas']:
                            if fig := analyze_tiff_statistics(str(ruta_blh_para_analisis), return_fig=params['show_plots']): params['fig_queue'].put(fig)
                        if params['generar_mapas']:
                            if fig := generar_mapa_con_leyenda(ruta_blh_para_analisis, aoi_path, params['title_date'], params['year'], cmap='turbo', producto="Altura de Capa Límite (BLH)", unidad="(m)", return_fig=params['show_plots']): params['fig_queue'].put(fig)
            
            else: 
                ruta_blh_para_calculo = None
    
    # --- CHEQUEO DE CANCELACIÓN ---
    if cancel_event.is_set(): print("--- 🛑 Cancelado después del paso BLH. ---"); return ruta_no2_para_procesar, None, None

    # --- 3. CÁLCULO DE CONCENTRACIÓN ---
    # Verificamos si podemos proceder al cálculo
    can_calculate = False
    if ruta_no2_para_procesar and params.get('transform_method'):
        if params['transform_method'] == 'petetin':
            can_calculate = (ruta_blh_para_calculo is not None)
        else:
            can_calculate = True # Savanets o Custom no necesitan archivo BLH
    
    if can_calculate:
        # Generar nombre de carpeta de salida
        # --- NUEVO: Lógica de nombres de carpeta según método ---
        if params['transform_method'] == 'savanets':
             folder_leaf = "Savenets Equation"
        elif params['transform_method'] == 'custom':
             val = params.get('transform_value', 0)
             # Formatear bonito (sin decimales si es entero)
             val_str = f"{int(val)}" if val == int(val) else f"{val}"
             folder_leaf = f"{val_str} transformed"
        else:
             # Caso Petetin o Default
             folder_suffix = params['output_name_blh'] if params.get('output_name_blh') else params['output_name']
             folder_leaf = folder_suffix.replace('BLH', 'Concentracion').replace('Datos_NO2', 'Concentracion')
        
        calc_output_dir = BASE_OUTPUT_PATH / f"Calculos/{params['year']}/{region_nombre}/{folder_leaf}"
        # --------------------------------------------------------
        calc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determinar si corresponde agregar el sufijo _regrid
        file_suffix = ""
        if params['do_regrid'] and ruta_no2_para_procesar != ruta_crudo_no2:
            file_suffix = "_regrid"

        ruta_concentracion_final = procesar_concentracion_no2(
            ruta_no2_para_procesar, 
            ruta_blh_para_calculo, # Puede ser None si no es Petetin
            calc_output_dir, 
            region_nombre, 
            params['year'], 
            params['title_date'],
            params['formato_salida'],
            metodo_transform=params['transform_method'],
            valor_custom=params.get('transform_value'),
            suffix=file_suffix
        )
        
        if ruta_concentracion_final:
            if not params.get('do_comparative_map', False):
                if params['estadisticas']:
                    if fig := analyze_tiff_statistics(str(ruta_concentracion_final), return_fig=params['show_plots']): params['fig_queue'].put(fig)
                if params['generar_mapas']:
                    if fig := generar_mapa_con_leyenda(ruta_concentracion_final, aoi_path, params['title_date'], params['year'], cmap='inferno', producto="Concentración de NO₂", unidad="(ppb)", return_fig=params['show_plots']): params['fig_queue'].put(fig)
    elif params.get('transform_method') == 'petetin':
        print(f"\n⚠️  Advertencia: Se omitió el cálculo de concentración porque falta el archivo de NO₂ o de BLH.")

    # --- CHEQUEO DE CANCELACIÓN ---
    if cancel_event.is_set(): print("--- 🛑 Cancelado después del cálculo de concentración. ---"); return ruta_no2_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final

    # --- 4. COMPRESIÓN ---
    if params['do_zip'] and ruta_crudo_no2:
        source_folder_to_zip = Path(request_no2.data_folder).resolve()
        zip_destino = source_folder_to_zip.parent / f"{source_folder_to_zip.name}_NO2-crudo.zip"
        print(f"\n🔄 Comprimiendo datos crudos de NO₂ en {zip_destino}...")
        comprimir_directorio(source_folder_to_zip, zip_destino)
        print("✅ Compresión finalizada.")

    # --- CHEQUEO DE CANCELACIÓN ---
    if cancel_event.is_set(): print("--- 🛑 Cancelado después de la compresión. ---"); return ruta_no2_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final

    # --- 5. RETORNO DE RUTAS ---
    return ruta_no2_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final

# ==============================================================================
# SECCIÓN 2: CLASE DE LA APLICACIÓN TKINTER
# ==============================================================================
class GeoApp:
    def __init__(self, root, available_regions):
        self.root = root
        self.root.title("Descargador y Procesador de Datos Atmosféricos")
        self.root.geometry("850x920") # Aumentado ligeramente para los nuevos controles
        self.fig_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.available_regions = available_regions
        
        # --- NUEVO: Evento para manejar la cancelación ---
        self.cancel_event = threading.Event()
        
        # Tamaños estándar para widgets
        self.ancho_spinbox_ano = 6
        self.ancho_combobox_mes = 12

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_date_widgets(main_frame)
        self.create_region_widgets(main_frame)
        self.create_options_widgets(main_frame)
        self.create_console_widgets(main_frame)
        self.create_action_buttons(main_frame)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.check_fig_queue)
        
        self.toggle_date_widgets()

    def on_closing(self):
        print("\nCerrando la aplicación...")
        if self.cancel_event:
            self.cancel_event.set() # Señaliza al thread que se detenga
        sys.stdout = self.original_stdout
        self.root.destroy()
        
    def create_date_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="1. Selección de Fecha", padding="10")
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.date_choice = tk.StringVar(value="mes")
        
        radio_frame = ttk.Frame(frame)
        radio_frame.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(radio_frame, text="Mes Particular", variable=self.date_choice, value="mes", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Año Completo", variable=self.date_choice, value="anio", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Día Puntual", variable=self.date_choice, value="dia", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Rango Días", variable=self.date_choice, value="rango", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Rango Meses", variable=self.date_choice, value="rango_meses", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)

        
        self.date_widgets_frame = ttk.Frame(frame)
        self.date_widgets_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)
        
        self.mes_label = ttk.Label(self.date_widgets_frame, text="Mes:")
        self.mes_combo = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes)
        self.mes_combo.set("Enero")
        self.ano_label = ttk.Label(self.date_widgets_frame, text="Año:")
        self.ano_spin = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        
        self.ano_label_completo = ttk.Label(self.date_widgets_frame, text="Año:")
        self.ano_spin_completo = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)

        self.dia_label = ttk.Label(self.date_widgets_frame, text="Fecha:")
        self.dia_cal = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        
        self.rango_label1 = ttk.Label(self.date_widgets_frame, text="Desde:")
        self.rango_cal1 = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_label2 = ttk.Label(self.date_widgets_frame, text="Hasta:")
        self.rango_cal2 = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        
        self.rango_mes_label_ini = ttk.Label(self.date_widgets_frame, text="Desde:")
        self.rango_mes_combo_ini = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes)
        self.rango_mes_combo_ini.set("Enero")
        self.rango_ano_spin_ini = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        
        self.rango_mes_label_fin = ttk.Label(self.date_widgets_frame, text="Hasta:")
        self.rango_mes_combo_fin = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes)
        self.rango_mes_combo_fin.set("Marzo")
        self.rango_ano_spin_fin = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)


    def toggle_date_widgets(self):
        for widget in self.date_widgets_frame.winfo_children():
            widget.grid_forget()
        
        choice = self.date_choice.get()
        
        if choice == "mes":
            self.mes_label.grid(row=0, column=0, sticky="w", padx=5)
            self.mes_combo.grid(row=0, column=1, sticky="w", padx=5)
            self.ano_label.grid(row=0, column=2, sticky="w", padx=5)
            self.ano_spin.grid(row=0, column=3, sticky="w", padx=5)
        elif choice == "anio":
            self.ano_label_completo.grid(row=0, column=0, sticky="w", padx=5)
            self.ano_spin_completo.grid(row=0, column=1, sticky="w", padx=5)
        elif choice == "dia":
            self.dia_label.grid(row=0, column=0, sticky="w", padx=5)
            self.dia_cal.grid(row=0, column=1, sticky="w", padx=5)
        elif choice == "rango":
            self.rango_label1.grid(row=0, column=0, sticky="w", padx=5)
            self.rango_cal1.grid(row=0, column=1, sticky="w", padx=5)
            self.rango_label2.grid(row=0, column=2, sticky="w", padx=5)
            self.rango_cal2.grid(row=0, column=3, sticky="w", padx=5)
        elif choice == "rango_meses":
            self.rango_mes_label_ini.grid(row=0, column=0, sticky="w", padx=5)
            self.rango_mes_combo_ini.grid(row=0, column=1, sticky="w", padx=5)
            self.rango_ano_spin_ini.grid(row=0, column=2, sticky="w", padx=5)
            
            self.rango_mes_label_fin.grid(row=1, column=0, sticky="w", padx=5, pady=(5,0))
            self.rango_mes_combo_fin.grid(row=1, column=1, sticky="w", padx=5, pady=(5,0))
            self.rango_ano_spin_fin.grid(row=1, column=2, sticky="w", padx=5, pady=(5,0))

        self.update_options_state()

    def create_region_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Selección de Región", padding="10")
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.region_mode = tk.StringVar(value="list")
        
        # Radio buttons para selección de modo
        rb_frame = ttk.Frame(frame)
        rb_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(rb_frame, text="Seleccionar de Lista Precargada", variable=self.region_mode, value="list", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(rb_frame, text="Coordenadas Manuales (BBox)", variable=self.region_mode, value="manual", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        
        # Contenedor para los sub-frames
        self.region_container = ttk.Frame(frame)
        self.region_container.pack(fill=tk.X, expand=True, pady=5)
        
        # --- MODO LISTA (Combobox existente) ---
        self.frame_list = ttk.Frame(self.region_container)
        ttk.Label(self.frame_list, text="Región:").pack(side=tk.LEFT, padx=5)
        self.region_combo = ttk.Combobox(self.frame_list, values=self.available_regions, state="readonly", width=40)
        self.region_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        if self.available_regions:
            default_region = "region_metropolitana_de_santiago"
            if default_region in self.available_regions:
                self.region_combo.set(default_region)
            else:
                self.region_combo.set(self.available_regions[0])
            
        # --- MODO MANUAL (Entradas de Coordenadas) ---
        self.frame_manual = ttk.Frame(self.region_container)
        
        # Nombre de la región manual
        nm_frame = ttk.Frame(self.frame_manual)
        nm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(nm_frame, text="Nombre de Zona (sin espacios):").pack(side=tk.LEFT)
        self.manual_name_var = tk.StringVar()
        ttk.Entry(nm_frame, textvariable=self.manual_name_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # Grid de coordenadas
        coord_frame = ttk.Frame(self.frame_manual)
        coord_frame.pack(fill=tk.X, pady=5)
        
        # Variables para coords
        self.min_lon = tk.DoubleVar(value=-71.0)
        self.min_lat = tk.DoubleVar(value=-33.6)
        self.max_lon = tk.DoubleVar(value=-70.0)
        self.max_lat = tk.DoubleVar(value=-33.0)
        
        # Etiquetas y Entradas (Min Lon, Min Lat, Max Lon, Max Lat)
        ttk.Label(coord_frame, text="Min Longitud (W):").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(coord_frame, textvariable=self.min_lon, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(coord_frame, text="Max Longitud (E):").grid(row=0, column=2, padx=5, pady=2, sticky="e")
        ttk.Entry(coord_frame, textvariable=self.max_lon, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(coord_frame, text="Min Latitud (S):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(coord_frame, textvariable=self.min_lat, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(coord_frame, text="Max Latitud (N):").grid(row=1, column=2, padx=5, pady=2, sticky="e")
        ttk.Entry(coord_frame, textvariable=self.max_lat, width=10).grid(row=1, column=3, padx=5, pady=2)
        
        # Inicializar estado visual
        self.toggle_region_mode()

    def toggle_region_mode(self):
        """Muestra u oculta los frames de selección según el modo elegido"""
        self.frame_list.pack_forget()
        self.frame_manual.pack_forget()
        
        if self.region_mode.get() == "list":
            self.frame_list.pack(fill=tk.X, expand=True)
        else:
            self.frame_manual.pack(fill=tk.X, expand=True)

    def create_options_widgets(self, parent):
        proc_frame = ttk.LabelFrame(parent, text="3. Opciones de Procesamiento", padding="10")
        proc_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.do_stats = tk.BooleanVar(value=True)
        self.do_maps = tk.BooleanVar(value=True)
        self.do_transform = tk.BooleanVar(value=True) # Renombrado de do_blh
        self.do_zip = tk.BooleanVar()
        self.do_regrid = tk.BooleanVar(value=False)
        self.do_comparative_map = tk.BooleanVar(value=False)
        
        # --- MODIFICACIÓN DE LA UI PARA TRANSFORMACIÓN ---
        # 1. Checkbox con nuevo texto y comando para activar/desactivar dropdown
        self.transform_checkbutton = ttk.Checkbutton(
            proc_frame, 
            text="Transformar a superficie", 
            variable=self.do_transform,
            command=self.toggle_transform_options
        )
        self.transform_checkbutton.pack(anchor="w")
        
        # 2. Frame interno para indentar las opciones
        self.transform_options_frame = ttk.Frame(proc_frame)
        self.transform_options_frame.pack(anchor="w", padx=20)
        
        # 3. Combobox con las 3 opciones
        self.transform_method_var = tk.StringVar(value="Descargar BLH y Calcular Concentración (H. Petetin Mode)")
        self.transform_method_combo = ttk.Combobox(
            self.transform_options_frame, 
            textvariable=self.transform_method_var,
            values=[
                "Descargar BLH y Calcular Concentración (H. Petetin Mode)",
                "Ecuación de Savanets (10km de altitud)",
                "Valor escrito por el usuario en pantalla"
            ],
            state="readonly",
            width=50
        )
        self.transform_method_combo.pack(anchor="w")
        # -------------------------------------------------
        
        ttk.Checkbutton(proc_frame, text="Re-escalar datos NO₂ (Kriging)", variable=self.do_regrid).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Analizar Estadísticas", variable=self.do_stats).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Generar Mapas Individuales", variable=self.do_maps).pack(anchor="w")

        self.comp_map_checkbutton = ttk.Checkbutton(proc_frame, text="Generar Mapa Comparativo (sólo Año/Rango Meses)", variable=self.do_comparative_map)
        self.comp_map_checkbutton.pack(anchor="w")

        ttk.Checkbutton(proc_frame, text="Comprimir resultados de NO₂ al finalizar", variable=self.do_zip).pack(anchor="w")
        
        ttk.Separator(proc_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(proc_frame, text="Formato de Salida (Ráster):").pack(anchor='w', pady=(5,0))
        self.formato_salida_var = tk.StringVar(value="GeoTIFF")
        self.formato_combo = ttk.Combobox(proc_frame, textvariable=self.formato_salida_var, values=["GeoTIFF", "NetCDF4", "ASCII Grid (.asc)"], state="readonly")
        self.formato_combo.pack(fill=tk.X, anchor='w')
        
        vis_frame = ttk.LabelFrame(parent, text="4. Visualización", padding="10")
        vis_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.show_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Mostrar gráficos/mapas generados", variable=self.show_plots).pack(anchor="w")
        ttk.Separator(vis_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(vis_frame, text="Paleta de Colores (NO₂ Columna):").pack(anchor='w', pady=(5,0))
        self.palette_combo = ttk.Combobox(vis_frame, values=list(paletas_colores.keys()), state="readonly")
        self.palette_combo.pack(fill=tk.X); self.palette_combo.set("viridis")

        # --- Sección de Nubosidad debajo de la paleta ---
        ttk.Separator(vis_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(vis_frame, text="Verificación Rápida de Nubosidad:", font=("Default", 9, "bold")).pack(anchor='w', pady=(0, 5))
        
        cloud_frame = ttk.Frame(vis_frame)
        cloud_frame.pack(fill=tk.X)
        
        self.btn_cloud = ttk.Button(cloud_frame, text="☁️ Calcular % Nubes", command=self.on_cloud_click)
        self.btn_cloud.pack(side=tk.LEFT, padx=(0, 10))
        
        # Pantallita para mostrar el resultado
        self.lbl_cloud_result = ttk.Label(cloud_frame, text="--- %", background="black", foreground="#00ff00", font=("Courier", 12, "bold"), padding=5, width=10, anchor="center")
        self.lbl_cloud_result.pack(side=tk.LEFT)
        # -------------------------------------------------------

    # --- MÉTODOS PARA EL BOTÓN DE NUBES ---
    def on_cloud_click(self):
        """Maneja el clic del botón de nubosidad."""
        try:
            # Reutilizamos get_params para obtener fechas y región actuales
            params = self.get_params()
            if not params:
                return
            
            # Deshabilitar botón mientras calcula
            self.btn_cloud.config(state="disabled")
            self.lbl_cloud_result.config(text="Calc...", foreground="yellow")
            
            # Iniciar thread
            threading.Thread(target=self.run_cloud_check, args=(params,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar parámetros: {e}")
            self.btn_cloud.config(state="normal")
            self.lbl_cloud_result.config(text="Error", foreground="red")

    def run_cloud_check(self, params):
        """Ejecuta la función global de nubosidad y actualiza la UI."""
        try:
            # Extraer fechas y ruta
            t_start = params.get('start_date')
            t_end = params.get('end_date')
            route = params.get('aoi_path')
            
            if not t_start or not t_end or not route:
                self.update_cloud_ui("Datos Insuf.", "red")
                return

            # Llamada a la función global
            percent = calcular_estadisticas_nubosidad(t_start, t_end, route)
            
            if percent is not None:
                color = "#00ff00" if percent < 20 else ("orange" if percent < 50 else "red")
                text = f"{percent:.1f}%"
                self.update_cloud_ui(text, color)
            else:
                self.update_cloud_ui("Error", "red")
                
        except Exception as e:
            print(f"Error en thread nubosidad: {e}")
            self.update_cloud_ui("Error", "red")
    
    def update_cloud_ui(self, text, color):
        """Actualiza la etiqueta de resultado de forma segura (thread-safe)."""
        def _update():
            self.lbl_cloud_result.config(text=text, foreground=color)
            self.btn_cloud.config(state="normal")
        self.root.after(0, _update)
    # -----------------------------------------------
    
    def toggle_transform_options(self):
        """Habilita o deshabilita el combobox según el estado del checkbox"""
        if self.do_transform.get():
            self.transform_method_combo.config(state="readonly")
        else:
            self.transform_method_combo.config(state="disabled")

    def update_options_state(self):
        choice = self.date_choice.get()
        
        # --- MODIFICADO: Añadido 'dia' a la lista para habilitar la transformación ---
        if choice in ['mes', 'anio', 'rango_meses', 'dia']:
            self.transform_checkbutton.config(state="normal")
            if self.do_transform.get():
                self.transform_method_combo.config(state="readonly")
        else:
            self.transform_checkbutton.config(state="disabled")
            self.do_transform.set(False)
            self.transform_method_combo.config(state="disabled")

        if choice in ['anio', 'rango_meses']:
            self.comp_map_checkbutton.config(state="normal")
        else:
            self.comp_map_checkbutton.config(state="disabled")
            self.do_comparative_map.set(False)

    def create_console_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Consola de Salida", padding="10")
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        parent.grid_rowconfigure(3, weight=1); parent.grid_columnconfigure(0, weight=1)
        self.console = scrolledtext.ScrolledText(frame, state='disabled', height=18, wrap=tk.WORD, bg="black", fg="white", font=("Courier New", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        sys.stdout = self.TextRedirector(self)

    def create_action_buttons(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        self.run_button = ttk.Button(frame, text="Iniciar Proceso", command=self.start_processing_thread)
        self.run_button.pack(side=tk.RIGHT, padx=5)

        # --- NUEVO BOTÓN: Descargar todas las regiones ---
        self.btn_download_all = ttk.Button(frame, text="Descargar todas las regiones", command=self.start_batch_processing_thread)
        self.btn_download_all.pack(side=tk.RIGHT, padx=5)
        # -------------------------------------------------
        
        # --- Botón Cancelar ---
        self.cancel_button = ttk.Button(frame, text="Cancelar", command=self.request_cancellation, state="disabled")
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(frame, text="Salir", command=self.on_closing).pack(side=tk.RIGHT)

    def start_processing_thread(self):
        # --- Proceso Single ---
        self.cancel_event.clear()
        self.run_button.config(state="disabled")
        self.btn_download_all.config(state="disabled") # Deshabilitar batch btn
        self.cancel_button.config(state="normal")
        
        self.clear_console()
        print("--- INICIANDO PROCESO (Región Única) ---")
        try:
            params = self.get_params()
            if params:
                thread = threading.Thread(target=self.start_single_processing, args=(params, self.cancel_event))
                thread.daemon = True
                thread.start()
            else:
                self.processing_finished()
        except Exception as e:
            messagebox.showerror("Error de Configuración", str(e))
            self.processing_finished()

    def start_batch_processing_thread(self):
        # --- Proceso Batch ---
        self.cancel_event.clear()
        self.run_button.config(state="disabled")
        self.btn_download_all.config(state="disabled")
        self.cancel_button.config(state="normal")
        
        self.clear_console()
        print("--- INICIANDO PROCESO POR LOTES (Todas las Regiones) ---")
        
        # Obtenemos params base IGNORANDO la región seleccionada en el combo
        try:
            base_params = self.get_params(ignore_region=True)
            if base_params:
                thread = threading.Thread(target=self.start_batch_processing, args=(base_params, self.cancel_event))
                thread.daemon = True
                thread.start()
            else:
                self.processing_finished()
        except Exception as e:
            messagebox.showerror("Error de Configuración", str(e))
            self.processing_finished()

    def start_single_processing(self, params, cancel_event):
        try:
            self._execute_process_flow(params, cancel_event)
        finally:
            self.processing_finished()

    def start_batch_processing(self, base_params, cancel_event):
        try:
            total_regiones = len(self.available_regions)
            if total_regiones == 0:
                print("❌ No se encontraron regiones disponibles en la carpeta 'Regiones'.")
                return

            print(f"📋 Se encontraron {total_regiones} regiones para procesar.")
            
            for i, region_name in enumerate(self.available_regions):
                if cancel_event.is_set():
                    print("\n🛑 Proceso por lotes detenido por el usuario.")
                    break
                
                print(f"\n\n>>> 🌍 PROCESANDO REGIÓN {i+1}/{total_regiones}: {region_name} <<<")
                
                # Crear parámetros específicos para esta región
                current_params = base_params.copy()
                current_params['aoi_path'] = BASE_GEOJSON_PATH / f"{region_name}.geojson"
                
                # Ejecutar lógica principal
                self._execute_process_flow(current_params, cancel_event)
                
            if not cancel_event.is_set():
                print("\n\n✅✅ ¡PROCESAMIENTO DE TODAS LAS REGIONES COMPLETADO! ✅✅")
                
        except Exception as e:
            print(f"❌ Error crítico en el proceso por lotes: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing_finished()

    # --- Método para el botón Cancelar ---
    def request_cancellation(self):
        print("\n--- 🛑 SOLICITUD DE CANCELACIÓN RECIBIDA ---")
        print("--- El proceso se detendrá después de la tarea actual... ---")
        self.cancel_event.set()
        self.cancel_button.config(state="disabled") # Deshabilita para evitar doble click

    def _get_month_list(self, y1, m1, y2, m2):
        """Genera una lista de tuplas (año, mes) entre dos fechas."""
        months = []
        fecha_ini = datetime(y1, m1, 1)
        fecha_fin = datetime(y2, m2, 1)
        
        if fecha_ini > fecha_fin:
             raise ValueError("La fecha de inicio (Mes/Año) no puede ser posterior a la fecha de fin.")
             
        current_year = y1
        current_month = m1
        
        while True:
            months.append((current_year, current_month))
            
            if current_year == y2 and current_month == m2:
                break
                
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        return months

    def get_params(self, ignore_region=False):
        params = {}
        choice = self.date_choice.get()
        params['choice_mode'] = choice
        
        mes_num_map = {name: num for num, name in meses_dict.items()}
        
        if choice == "mes":
            mes_nombre = self.mes_combo.get()
            mes_num = mes_num_map[mes_nombre]
            ano = int(self.ano_spin.get())
            _, last_day = calendar.monthrange(ano, mes_num)
            params['start_date'] = f"{ano}-{mes_num:02d}-01"
            params['end_date'] = f"{ano}-{mes_num:02d}-{last_day}"
            params['year'], params['month'] = ano, mes_num
            params['title_date'] = meses_es_lower[mes_num]
            params['output_name'] = f"{mes_num:02d}_Datos_NO2_{params['title_date']}_{ano}"
            params['output_name_blh'] = f"{mes_num:02d}_Datos_BLH_{params['title_date']}_{ano}"
            params['month_list'] = [(ano, mes_num)]
            params['title_suffix'] = f"{mes_nombre} {ano}"
            
        elif choice == "dia":
            fecha = self.dia_cal.get_date().strftime('%Y-%m-%d')
            params['start_date'] = params['end_date'] = fecha
            params['year'] = int(fecha[:4])
            params['title_date'] = fecha
            params['output_name'] = f"Datos_NO2_Dia_{fecha}"
            params['title_suffix'] = f"Dia {fecha}"

        elif choice == "rango":
            start_date_obj = self.rango_cal1.get_date()
            end_date_obj = self.rango_cal2.get_date()
            if start_date_obj > end_date_obj:
                raise ValueError("La fecha de inicio no puede ser posterior a la fecha de fin.")
            start_date, end_date = start_date_obj.strftime('%Y-%m-%d'), end_date_obj.strftime('%Y-%m-%d')
            params['start_date'], params['end_date'] = start_date, end_date
            params['year'] = int(start_date[:4])
            params['title_date'] = f"Rango de {start_date} a {end_date}"
            params['output_name'] = f"Datos_NO2_Rango_{start_date}_a_{end_date}"
            params['title_suffix'] = f"Rango {start_date} a {end_date}"

        elif choice == "anio":
            ano = int(self.ano_spin_completo.get())
            params['year'] = ano
            params['month_list'] = [(ano, m) for m in range(1, 13)]
            params['title_suffix'] = f"Año {ano}"

        elif choice == "rango_meses":
            m1_nombre = self.rango_mes_combo_ini.get()
            m1 = mes_num_map[m1_nombre]
            y1 = int(self.rango_ano_spin_ini.get())
            
            m2_nombre = self.rango_mes_combo_fin.get()
            m2 = mes_num_map[m2_nombre]
            y2 = int(self.rango_ano_spin_fin.get())

            params['month_list'] = self._get_month_list(y1, m1, y2, m2)
            params['year'] = y1 
            params['title_suffix'] = f"Rango {m1_nombre} {y1} - {m2_nombre} {y2}"
            
        # --- LÓGICA DE SELECCIÓN DE REGIÓN (LISTA vs MANUAL) ---
        if not ignore_region:
            if self.region_mode.get() == "list":
                region_nombre_archivo = self.region_combo.get()
                if not region_nombre_archivo:
                     raise ValueError("Por favor, seleccione una región de la lista.")

                params['aoi_path'] = BASE_GEOJSON_PATH / f"{region_nombre_archivo}.geojson"
                if not params['aoi_path'].exists():
                    raise FileNotFoundError(f"No se encontró el archivo GeoJSON para la región seleccionada:\n{params['aoi_path']}")
            else:
                # Modo Manual: Crear GeoJSON temporal
                nombre_manual = self.manual_name_var.get().strip()
                if not nombre_manual:
                    raise ValueError("Por favor, ingrese un nombre para la zona manual.")
                nombre_manual = "".join([c for c in nombre_manual if c.isalnum() or c in (' ', '_', '-')]).replace(" ", "_")
                
                try:
                    min_x = self.min_lon.get()
                    min_y = self.min_lat.get()
                    max_x = self.max_lon.get()
                    max_y = self.max_lat.get()
                    
                    if min_x >= max_x or min_y >= max_y:
                        raise ValueError("Las coordenadas mínimas deben ser menores que las máximas.")
                    
                    bbox_geom = box(min_x, min_y, max_x, max_y)
                    gdf_manual = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="EPSG:4326")
                    
                    archivo_salida = BASE_GEOJSON_PATH / f"Manual_{nombre_manual}.geojson"
                    print(f"📝 Generando archivo de región manual: {archivo_salida}")
                    gdf_manual.to_file(archivo_salida, driver="GeoJSON")
                    
                    params['aoi_path'] = archivo_salida
                    
                except ValueError as ve:
                    raise ValueError(f"Error en coordenadas manuales: {ve}")
                except Exception as e:
                    raise RuntimeError(f"Error al crear el GeoJSON manual: {e}")
        else:
            params['aoi_path'] = None # Se rellenará en el bucle batch

        # Opciones generales
        params.update({
            "estadisticas": self.do_stats.get(),
            "generar_mapas": self.do_maps.get(), "do_zip": self.do_zip.get(),
            "show_plots": self.show_plots.get(), "palette": self.palette_combo.get(),
            "fig_queue": self.fig_queue,
            "formato_salida": self.formato_salida_var.get(),
            "do_regrid": self.do_regrid.get(),
            "do_comparative_map": self.do_comparative_map.get()
        })

        # --- LÓGICA DE TRANSFORMACIÓN (Parámetros) ---
        params['transform_method'] = None
        params['transform_value'] = None
        params['descargar_blh'] = False 

        if self.do_transform.get():
            selection = self.transform_method_combo.get()
            
            if "Petetin Mode" in selection:
                params['transform_method'] = "petetin"
                params['descargar_blh'] = True 
                
            elif "Ecuación de Savanets" in selection:
                params['transform_method'] = "savanets"
                
            elif "Valor escrito por el usuario" in selection:
                params['transform_method'] = "custom"
                val = simpledialog.askfloat(
                    "Valor de Altura de Capa de Mezcla", 
                    "Ingrese el valor del denominador (metros) para la transformación:\n(Ej: 500, 1000, 1500)",
                    parent=self.root,
                    minvalue=1.0, maxvalue=50000.0
                )
                if val is None:
                    print("Operación cancelada por el usuario (sin valor de altura).")
                    return None 
                params['transform_value'] = val
        
        return params

    def _execute_process_flow(self, params, cancel_event):
        """
        Ejecuta el flujo de procesamiento para una configuración dada.
        NOTA: Ya no restaura los botones al final (eso lo hace el caller).
        """
        try:
            if params.get('choice_mode') in ['anio', 'rango_meses']:
                
                print(f"\n==========================================================")
                print(f"🚀 INICIANDO PROCESO MULTI-MES PARA: {params['title_suffix']}")
                print(f"==========================================================")
                
                base_params = params.copy()
                months_to_process = base_params['month_list']
                
                no2_paths = []
                blh_paths = []
                concentracion_paths = []
                
                for year, month in months_to_process:
                    if cancel_event.is_set():
                        print("--- 🛑 Proceso multi-mes cancelado por el usuario. ---")
                        break 

                    params_mes = base_params.copy()
                    mes_nombre_es = meses_es_lower[month]
                    _, last_day = calendar.monthrange(year, month)
                    
                    params_mes['start_date'] = f"{year}-{month:02d}-01"
                    params_mes['end_date'] = f"{year}-{month:02d}-{last_day}"
                    params_mes['year'] = year
                    params_mes['month'] = month
                    params_mes['title_date'] = mes_nombre_es
                    params_mes['output_name'] = f"{month:02d}_Datos_NO2_{mes_nombre_es}_{year}"
                    
                    if params_mes['descargar_blh']:
                         params_mes['output_name_blh'] = f"{month:02d}_Datos_BLH_{mes_nombre_es}_{year}"
                    else:
                         params_mes['output_name_blh'] = f"Datos_Transformados_{params_mes.get('transform_method')}"
                    
                    no2_p, blh_p, conc_p = run_processing(params_mes, cancel_event)
                    
                    if no2_p: no2_paths.append(no2_p)
                    if blh_p: blh_paths.append(blh_p)
                    if conc_p: concentracion_paths.append(conc_p)
                    
                    if not cancel_event.is_set():
                        print(f"\n--- ✅ Mes {mes_nombre_es.capitalize()} {year} completado. ---")
                
                print(f"\n==========================================================")
                if cancel_event.is_set():
                    print(f"🎉 PROCESO MULTI-MES CANCELADO")
                else:
                    print(f"🎉 PROCESO MULTI-MES FINALIZADO")
                print(f"==========================================================")

                if params['do_comparative_map'] and not cancel_event.is_set():
                    print("\n--- Iniciando generación de Mapas Comparativos ---")
                    fig_queue = params['fig_queue'] if params['show_plots'] else None

                    if no2_paths:
                        if fig := generar_mapa_comparativo(no2_paths, params['aoi_path'], "NO₂ (Columna Troposférica)", "(mol/m²)", params['palette'], params['title_suffix'], return_fig=params['show_plots']):
                            if fig_queue: fig_queue.put(fig)
                    if blh_paths and params['descargar_blh']:
                        if fig := generar_mapa_comparativo(blh_paths, params['aoi_path'], "Altura de Capa Límite (BLH)", "(m)", 'turbo', params['title_suffix'], return_fig=params['show_plots']):
                            if fig_queue: fig_queue.put(fig)
                    if concentracion_paths and params.get('transform_method'):
                        if fig := generar_mapa_comparativo(concentracion_paths, params['aoi_path'], "Concentración de NO₂", "(ppb)", 'inferno', params['title_suffix'], return_fig=params['show_plots']):
                            if fig_queue: fig_queue.put(fig)

                if params['estadisticas'] and params['do_comparative_map'] and not cancel_event.is_set():
                    print("\n--- Iniciando generación de Estadísticas (para proceso multi-mes) ---")
                    all_paths = no2_paths + blh_paths + concentracion_paths
                    for p in all_paths:
                        if p: 
                            if fig := analyze_tiff_statistics(str(p), return_fig=params['show_plots']):
                                if params['show_plots']: params['fig_queue'].put(fig)

            else:
                run_processing(params, cancel_event)
            
        except Exception as e:
            print(f"\n❌ ERROR INESPERADO DURANTE LA EJECUCIÓN: {e}")
            import traceback
            traceback.print_exc()

    def processing_finished(self):
        # --- Restaura los botones a su estado original ---
        if self.cancel_event.is_set():
             print("\n\n🛑 ¡PROCESO DETENIDO!")
        else:
             print("\n\n✅ ¡TAREA FINALIZADA!")
        
        self.run_button.config(state="normal")
        self.btn_download_all.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.cancel_event.clear()

    def check_fig_queue(self):
        try:
            fig = self.fig_queue.get_nowait()
            self.display_figure(fig)
        except queue.Empty:
            pass
        self.root.after(100, self.check_fig_queue)

    def display_figure(self, fig):
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Visualizador de Gráficos")
        plot_window.geometry("1200x800")
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def clear_console(self):
        self.console.config(state='normal')
        self.console.delete('1.0', tk.END)
        self.console.config(state='disabled')

    class TextRedirector:
        def __init__(self, app): self.app = app
        def write(self, str_):
            try:
                self.app.console.config(state='normal'); self.app.console.insert(tk.END, str_)
                self.app.console.see(tk.END); self.app.console.config(state='disabled')
            except tk.TclError: pass
        def flush(self): pass

# ==============================================================================
# SECCIÓN 3: PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    if not BASE_GEOJSON_PATH.is_dir():
        messagebox.showerror("Error de Configuración", f"No se encontró la carpeta 'Regiones'.\n\nAsegúrate de que una carpeta llamada 'Regiones' con los archivos .geojson exista en el mismo directorio que el script:\n\n{SCRIPT_DIR}")
        sys.exit(1)
    
    available_regions = get_available_regions()
    # MODIFICADO: Ya no forzamos salida si no hay regiones, pues el usuario puede crear manuales.
    if not available_regions:
        print("Advertencia: No se encontraron regiones precargadas. Deberá usar el modo manual.")
        
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass # Para sistemas no-Windows
    
    root = tk.Tk()
    app = GeoApp(root, available_regions)
    root.mainloop()
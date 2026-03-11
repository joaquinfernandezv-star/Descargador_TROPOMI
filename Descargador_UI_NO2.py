# --- IMPORTACIONES ---
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import threading
import queue
import sys
import os
import shutil
from pathlib import Path
import calendar
import json
import zipfile
from datetime import datetime, timedelta

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
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    from rasterio.transform import from_origin
    from scipy.stats import norm, pearsonr, spearmanr
    import contextily as cx
    from shapely.geometry import box
    from sentinelhub import (
        SHConfig, CRS, BBox, DataCollection, MimeType,
        SentinelHubRequest
    )
    from pykrige.ok import OrdinaryKriging
    import cdsapi
    import rioxarray
    import xarray as xr
except ImportError as e:
    messagebox.showerror("Librería Faltante", f"Falta una librería necesaria: {e}.\n\nPor favor, instálala usando 'pip install <libreria>'.")
    sys.exit(1)

# ==============================================================================
# SECCIÓN 1: LÓGICA DE PROCESO Y CONFIGURACIÓN
# ==============================================================================

# --- RUTAS DINÁMICAS ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_OUTPUT_PATH = SCRIPT_DIR / "Resultados"
BASE_GEOJSON_PATH = SCRIPT_DIR / "Regiones"

BASE_OUTPUT_PATH.mkdir(exist_ok=True)
BASE_GEOJSON_PATH.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DE GASES ---
GAS_CONFIG = {
    "NO2": { "nombre_corto": "NO2", "nombre_largo": "Dióxido de Nitrógeno", "band_name": "NO2", "peso_molecular": 46.01, "factor_ppb": 24.45, "comportamiento_vertical": "PBL_confined" },
    "O3": { "nombre_corto": "O3", "nombre_largo": "Ozono", "band_name": "O3", "peso_molecular": 48.00, "factor_ppb": 24.45, "comportamiento_vertical": "Thick_troposphere" },
    "SO2": { "nombre_corto": "SO2", "nombre_largo": "Dióxido de Azufre", "band_name": "SO2", "peso_molecular": 64.07, "factor_ppb": 24.45, "comportamiento_vertical": "PBL_confined" },
    "CO": { "nombre_corto": "CO", "nombre_largo": "Monóxido de Carbono", "band_name": "CO", "peso_molecular": 28.01, "factor_ppb": 24.45, "comportamiento_vertical": "Thick_troposphere" },
    "CH4": { "nombre_corto": "CH4", "nombre_largo": "Metano", "band_name": "CH4", "peso_molecular": 16.04, "factor_ppb": 24.45, "comportamiento_vertical": "Thick_troposphere" },
    "HCHO": { "nombre_corto": "HCHO", "nombre_largo": "Formaldehído", "band_name": "HCHO", "peso_molecular": 30.03, "factor_ppb": 24.45, "comportamiento_vertical": "PBL_confined" }
}

GASES_SUB_BLH = {k: v for k, v in GAS_CONFIG.items() if v["comportamiento_vertical"] == "PBL_confined"}
GASES_POST_BLH = {k: v for k, v in GAS_CONFIG.items() if v["comportamiento_vertical"] == "Thick_troposphere"}

# --- CONFIGURACIÓN DE SENTINEL HUB ---
try:
    config = SHConfig()
    config.download_timeout_seconds = 120 
    if not config.sh_client_id or not config.sh_client_secret:
        print("Configurando credenciales de Copernicus Data Space...")
        config.sh_client_id = "sh-2279fd56-dabb-4e4d-ae5b-b71ce5fc5c09"
        config.sh_client_secret = "9c94Zs5JMkwIkwGqyBJGCSXigh9jslVP"
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.save("cdse")
except Exception as e:
    print(f"Error configurando SentinelHub: {e}")

data_5p = DataCollection.SENTINEL5P.define_from("5p", service_url=config.sh_base_url)

# --- Evalscripts Dinámicos ---
def get_evalscript_gas(band_name):
    return f"""
    //VERSION=3
    function setup() {{ return {{ input: ["{band_name}"], output: {{ bands: 1, sampleType: "FLOAT32" }}, mosaicking: "SIMPLE" }}; }}
    function evaluatePixel(samples) {{ return [samples.{band_name}]; }}
    """

evalscript_cloud = """
//VERSION=3
function setup() { return { input: ["CLOUD_FRACTION"], output: { bands: 1, sampleType: "FLOAT32" }, mosaicking: "SIMPLE" }; }
function evaluatePixel(samples) { return [samples.CLOUD_FRACTION]; }
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
    'viridis': 'Púrpura a amarillo (Estándar)', 'plasma': 'Púrpura a amarillo cálido', 
    'inferno': 'Negro, rojo, amarillo', 'magma': 'Negro, púrpura, amarillo',
    'cividis': 'Azul a amarillo (Daltonismo)', 'turbo': 'Arcoiris mejorado',
    'Spectral': 'Rojo a Azul (Divergente)', 'RdYlGn': 'Rojo, Amarillo, Verde'
}
if tiene_cmcrameri: paletas_colores['batlow'] = 'Perceptual uniforme (Batlow)'

# --- FUNCIONES AUXILIARES ---
def get_available_regions():
    if not BASE_GEOJSON_PATH.is_dir(): return []
    return sorted([f.stem for f in BASE_GEOJSON_PATH.glob("*.geojson")])

def geojson_to_coords(geojson_path: str):
    try:
        gdf = gpd.read_file(geojson_path)
        bounds = gdf.total_bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]]
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

# --- CÁLCULO DE NUBOSIDAD ---
def calcular_estadisticas_nubosidad(time_start, time_end, route):
    print(f"☁️  Analizando nubosidad para {time_start} - {time_end}...")
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None

    request_cloud = SentinelHubRequest(
        evalscript=evalscript_cloud,
        input_data=[SentinelHubRequest.input_data(data_collection=data_5p, time_interval=(time_start, time_end), other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '50', 'timeliness': 'OFFL'}})],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)),
        resolution=(2000, 2000), config=config
    )
    
    try:
        data = request_cloud.get_data()[0] 
        cloud_values = data.flatten()
        valid_clouds = cloud_values[~np.isnan(cloud_values)]
        if valid_clouds.size == 0: return 0.0
        return np.mean(valid_clouds) * 100
    except Exception as e:
        print(f"❌ Error calculando nubosidad: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---
def descargar_y_promediar_gas(time_start, time_end, route, output_name, gas_info, cancel_event, generar_serie=False):
    area = Path(route).stem
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None, None, None, [], []
    
    gas_name = gas_info["nombre_corto"]
    band_name = gas_info["band_name"]
    evalscript_dinamico = get_evalscript_gas(band_name)
    
    year_folder = time_start[:4]
    final_folder = BASE_OUTPUT_PATH / f"Modelo_{gas_name}/{year_folder}/{area}/{output_name}"
    final_folder.mkdir(parents=True, exist_ok=True)
    final_tiff_path = final_folder / "response.tiff"
    csv_path = None
    
    start_dt = datetime.strptime(time_start, '%Y-%m-%d')
    end_dt = datetime.strptime(time_end, '%Y-%m-%d')
    delta = end_dt - start_dt

    # Lógica 1 día
    if delta.days == 0:
        request = SentinelHubRequest(
            evalscript=evalscript_dinamico,
            input_data=[SentinelHubRequest.input_data(data_collection=data_5p, time_interval=(time_start, time_end), other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '75', 'timeliness': 'OFFL'}})],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)), resolution=(5500, 3500), config=config, data_folder=str(final_folder)
        )
        try:
            request.get_data(save_data=True)
            filenames = request.get_filename_list()
            if filenames:
                tiff_p = (final_folder / filenames[0]).resolve()
                if generar_serie:
                    gdf = gpd.read_file(route).dissolve()
                    try:
                        with rasterio.open(tiff_p) as src:
                            gdf_proj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
                            data, _ = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                            val = data[0].flatten()
                            val = val[~np.isnan(val)]
                            if val.size > 0:
                                p5, p50, p95 = np.percentile(val, [5, 50, 95])
                                csv_path = final_folder / f"Serie_Temporal_Rango_{gas_name}_{time_start}.csv"
                                pd.DataFrame([{'Fecha': time_start, 'P5': p5, 'P50': p50, 'P95': p95}]).to_csv(csv_path, index=False)
                    except Exception: pass
                return tiff_p, csv_path, None, [tiff_p], [time_start]
        except Exception: pass
        return None, None, None, [], []

    # Lógica Multi-días
    temp_dir = BASE_OUTPUT_PATH / f"Temp_Daily_{gas_name}" / output_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    daily_tiffs = []
    daily_dates = []
    print(f"🔄 Iniciando descarga diaria de {gas_name} progresiva ({delta.days + 1} días)...")
    
    for i in range(delta.days + 1):
        if cancel_event.is_set():
            print("\n🛑 Descarga cancelada por el usuario.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, [], []
            
        current_date = (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
        day_folder = temp_dir / current_date
        day_folder.mkdir(exist_ok=True)
        
        sys.stdout.write(f"\r      Descargando {gas_name} día: {current_date}... ")
        sys.stdout.flush()
        
        request = SentinelHubRequest(
            evalscript=evalscript_dinamico,
            input_data=[SentinelHubRequest.input_data(data_collection=data_5p, time_interval=(current_date, current_date), other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '75', 'timeliness': 'OFFL'}})],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)), resolution=(5500, 3500), config=config, data_folder=str(day_folder)
        )
        try:
            request.get_data(save_data=True)
            filenames = request.get_filename_list()
            if filenames: 
                daily_tiffs.append(day_folder / filenames[0])
                daily_dates.append(current_date)
        except Exception: pass

    print(f"\n✅ Descargas de {gas_name} finalizadas. Promediando datos...")

    # Extraer Serie Temporal Mediana y Rango Diario si fue solicitado
    if generar_serie and daily_tiffs:
        stats_records = []
        gdf = gpd.read_file(route).dissolve()
        for tiff, d_str in zip(daily_tiffs, daily_dates):
            try:
                with rasterio.open(tiff) as src:
                    gdf_proj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
                    data, _ = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    val = data[0].flatten()
                    val = val[~np.isnan(val)]
                    if val.size > 0:
                        p5, p50, p95 = np.percentile(val, [5, 50, 95])
                        stats_records.append({'Fecha': d_str, 'P5': p5, 'P50': p50, 'P95': p95})
            except Exception: pass
        
        if stats_records:
            csv_path = final_folder / f"Serie_Temporal_Rango_{gas_name}.csv"
            pd.DataFrame(stats_records).to_csv(csv_path, index=False)

    if not daily_tiffs:
        print(f"❌ No se encontraron datos válidos de {gas_name} en el período.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, [], []

    try:
        datasets = []
        for tiff in daily_tiffs:
            try:
                ds = rioxarray.open_rasterio(tiff)
                nodata = ds.rio.nodata
                if nodata is not None: ds = ds.where(ds != nodata)
                datasets.append(ds)
            except Exception: pass
                
        if not datasets:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, [], []
            
        combined = xr.concat(datasets, dim='time')
        mean_ds = combined.mean(dim='time', skipna=True)
        
        ref_ds = datasets[0]
        mean_ds.rio.write_crs(ref_ds.rio.crs, inplace=True)
        if ref_ds.rio.nodata is not None:
            mean_ds = mean_ds.fillna(ref_ds.rio.nodata)
            mean_ds.rio.write_nodata(ref_ds.rio.nodata, inplace=True)
            
        mean_ds.rio.to_raster(final_tiff_path)
        print(f"✅ Promedio local de {gas_name} guardado con éxito.")
    except Exception as e:
        print(f"❌ Error al procesar el promedio: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, [], []
        
    return final_tiff_path.resolve(), csv_path, temp_dir, daily_tiffs, daily_dates

def extraer_serie_ppb(daily_tiffs, daily_dates, ruta_blh_crudo, metodo_transform, valor_custom, gas_info, ruta_geojson, output_csv):
    """Calcula matemáticamente el PPB diario iterando sobre cada imagen descargada y exporta sus percentiles."""
    if not daily_tiffs: return None
    try:
        gdf = gpd.read_file(ruta_geojson).dissolve()
        
        with rasterio.open(daily_tiffs[0]) as src_gas_ref:
            gas_meta = src_gas_ref.meta.copy()
            gdf_proj = gdf.to_crs(src_gas_ref.crs) if gdf.crs != src_gas_ref.crs else gdf

        val_blh_cropped = None
        if metodo_transform == "petetin" and ruta_blh_crudo:
            with rasterio.open(ruta_blh_crudo) as src_blh:
                h_resampled_full = np.empty((gas_meta['height'], gas_meta['width']), dtype=np.float32)
                reproject(
                    source=rasterio.band(src_blh, 1), destination=h_resampled_full,
                    src_transform=src_blh.transform, src_crs=src_blh.crs,
                    dst_transform=gas_meta['transform'], dst_crs=gas_meta['crs'],
                    resampling=Resampling.bilinear
                )
                h_resampled_full[h_resampled_full <= 0] = np.nan
                
            # Generar máscara geométrica rigurosa alineada al TIFF de gas usando un archivo en memoria
            with MemoryFile() as memfile:
                mem_meta = gas_meta.copy()
                mem_meta.update(dtype=rasterio.float32, count=1)
                with memfile.open(**mem_meta) as dataset:
                    dataset.write(h_resampled_full, 1)
                    data_blh, _ = mask(dataset=dataset, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    val_blh_cropped = data_blh[0]
                    
        elif metodo_transform == "savanets": val_blh_cropped = 10000.0
        elif metodo_transform == "custom": val_blh_cropped = float(valor_custom)
        else: return None

        # Factor multiplicador agrupa la conversión (24.45 / PesoMolecular) * (1e6 * PesoMolecular) = 24.45 * 1e6
        factor_multiplicador = gas_info["peso_molecular"] * 1e6 * (gas_info["factor_ppb"] / gas_info["peso_molecular"])

        stats_records = []
        for tiff, d_str in zip(daily_tiffs, daily_dates):
            try:
                with rasterio.open(tiff) as src:
                    data_gas, _ = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    val_gas = data_gas[0]
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        concentracion_ppb = (val_gas / val_blh_cropped) * factor_multiplicador
                        
                    val_flat = concentracion_ppb.flatten()
                    val_flat = val_flat[~np.isnan(val_flat)]
                    
                    if val_flat.size > 0:
                        p5, p50, p95 = np.percentile(val_flat, [5, 50, 95])
                        stats_records.append({'Fecha': d_str, 'P5': p5, 'P50': p50, 'P95': p95})
            except Exception: pass

        if stats_records:
            pd.DataFrame(stats_records).to_csv(output_csv, index=False)
            return output_csv
    except Exception as e: print(f"❌ Error extrayendo serie temporal PPB: {e}")
    return None

def descargar_blh_era5(year, month, area_coords, output_path):
    print("🛰️  Conectando al Copernicus CDS para descargar BLH (Promedio Mensual)...")
    c = cdsapi.Client()
    request_payload = {
        "product_type": ["monthly_averaged_reanalysis"], "variable": ["boundary_layer_height"],
        "year": [str(year)], "month": [f"{month:02d}"], "time": ["00:00"], 
        "data_format": "netcdf", "download_format": "unarchived", "area": area_coords
    }
    try:
        c.retrieve('reanalysis-era5-single-levels-monthly-means', request_payload, output_path)
        print(f"✅ Descarga BLH Mensual completada: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error durante la descarga de BLH mensual desde CDS: {e}")
        return False

def descargar_blh_era5_diario(start_date, end_date, area_coords, output_path):
    print(f"📡 Solicitando ERA5 Horario (12:00) para {start_date}...")
    c = cdsapi.Client()
    request_payload = {
        "product_type": ["reanalysis"], "variable": ["boundary_layer_height"],
        "year": [str(start_date.year)], "month": [f"{start_date.month:02d}"], "day": [f"{start_date.day:02d}"],
        "time": ["12:00"], "data_format": "netcdf", "download_format": "unarchived", "area": area_coords
    }
    try:
        c.retrieve('reanalysis-era5-single-levels', request_payload, output_path)
        print("✅ Descarga BLH Diario completada.")
        return True
    except Exception as e:
        print(f"❌ Error ERA5 Diario: {e}")
        return False

def convertir_nc_a_tiff(netcdf_path, tiff_path):
    try:
        data_array = rioxarray.open_rasterio(netcdf_path, variable='blh').squeeze()
        data_array.rio.write_crs("EPSG:4326", inplace=True)
        data_array.rio.to_raster(tiff_path, driver='GTiff')
        return True
    except Exception: return False

def tag_nc_with_crs(netcdf_path, output_nc_path):
    try:
        data_array = rioxarray.open_rasterio(netcdf_path, variable='blh').squeeze()
        data_array.rio.write_crs("EPSG:4326", inplace=True)
        data_array.rio.to_raster(output_nc_path, driver='NETCDF')
        return True
    except Exception: return False

def procesar_concentracion_gas(ruta_crudo, ruta_blh_crudo, output_dir, region_nombre, año, mes_nombre_es, gas_info, formato_salida="GeoTIFF", metodo_transform="petetin", valor_custom=None, suffix=""):
    gas_name = gas_info["nombre_corto"]
    print(f"\n---\n🔬 Calculando concentración de {gas_name} para: {mes_nombre_es} {año} (Modo: {metodo_transform})")
    
    try:
        with rasterio.open(ruta_crudo) as src_gas:
            profile, ccol_array = src_gas.profile, src_gas.read(1).astype(np.float32)
            if src_gas.nodata is not None: ccol_array[ccol_array == src_gas.nodata] = np.nan
            h_resampled = None

            if metodo_transform == "petetin":
                if not ruta_blh_crudo: return None
                with rasterio.open(ruta_blh_crudo) as src_blh:
                    h_resampled = np.empty_like(ccol_array)
                    reproject(source=rasterio.band(src_blh, 1), destination=h_resampled, src_transform=src_blh.transform, src_crs=src_blh.crs, dst_transform=src_gas.transform, dst_crs=src_gas.crs, resampling=Resampling.bilinear)
                    h_resampled[h_resampled <= 0] = np.nan
            elif metodo_transform == "savanets": h_resampled = 10000.0
            elif metodo_transform == "custom":
                if valor_custom is None: return None
                h_resampled = float(valor_custom)
            else: return None
            
            M, A, PPB_FACTOR = gas_info["peso_molecular"], 1e6, gas_info["factor_ppb"] / gas_info["peso_molecular"]
            with np.errstate(divide='ignore', invalid='ignore'):
                concentracion_ug_m3 = (ccol_array / h_resampled) * M * A
                concentracion_ppb = concentracion_ug_m3 * PPB_FACTOR
                
            if profile.get('nodata') is not None: concentracion_ppb[np.isnan(concentracion_ppb)] = profile['nodata']

            nombre_archivo_base = f"Concentracion_ppb_{gas_name}_{region_nombre}_{año}_{mes_nombre_es}{suffix}"

            if formato_salida == "NetCDF4":
                template_raster = rioxarray.open_rasterio(ruta_crudo)
                data_con_banda = concentracion_ppb.astype(np.float32)[np.newaxis, :, :]
                data_array = template_raster.copy(data=data_con_banda).rename(f'concentracion_{gas_name.lower()}_ppb')
                if profile.get('nodata') is not None: data_array = data_array.rio.write_nodata(profile['nodata'])
                output_path = output_dir / f"{nombre_archivo_base}.nc"
                data_array.rio.to_raster(output_path, driver="NETCDF")
            elif formato_salida == "ASCII Grid (.asc)":
                output_path = output_dir / f"{nombre_archivo_base}.asc"
                nodata_value = -9999.0
                concentracion_para_asc = np.nan_to_num(concentracion_ppb, nan=nodata_value)
                profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=nodata_value)
                with rasterio.open(output_path, 'w', **profile) as dst: dst.write(concentracion_para_asc.astype(rasterio.float32), 1)
            else: 
                output_path = output_dir / f"{nombre_archivo_base}.tiff"
                profile.update(dtype=rasterio.float32)
                with rasterio.open(output_path, 'w', **profile) as dst: dst.write(concentracion_ppb.astype(rasterio.float32), 1)

            print(f"✅ Cálculo de concentración completado: {output_path}")
            return output_path
    except Exception as e:
        print(f"❌ Error procesando concentración de {gas_name}: {e}")
        return None

# --- CÁLCULO DE RATIO Y CORRELACIÓN MATRICIAL ENTRE GASES ---
def calcular_y_guardar_ratio(ruta_num, ruta_den, output_dir, gas_num, gas_den, region, año, mes_str):
    try:
        with rasterio.open(ruta_num) as src_num, rasterio.open(ruta_den) as src_den:
            data_num = src_num.read(1).astype(np.float32)
            data_den = src_den.read(1).astype(np.float32)
            profile = src_num.profile

            # Manejo de NoData
            if src_num.nodata is not None: data_num[data_num == src_num.nodata] = np.nan
            if src_den.nodata is not None: data_den[data_den == src_den.nodata] = np.nan

            # Evitar división por cero y valores anómalos (negativos en VCD)
            data_den[data_den <= 0] = np.nan
            data_num[data_num < 0] = np.nan

            # Cálculo de la relación (Ratio)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_data = data_num / data_den

            output_path = output_dir / f"Ratio_{gas_num}_sobre_{gas_den}_{region}_{año}_{mes_str}.tiff"
            profile.update(dtype=rasterio.float32, nodata=np.nan)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(ratio_data, 1)

            print(f"✅ Mapa de relación {gas_num}/{gas_den} generado exitosamente.")
            return output_path
    except Exception as e:
        print(f"❌ Error al calcular Ratio: {e}")
        return None

def calcular_y_graficar_correlacion(ruta_num, ruta_den, ruta_geojson, gas_num, gas_den, region_nombre, año, mes_str, output_dir, return_fig=False):
    """Calcula Pearson y Spearman y genera un gráfico de densidad entre dos gases"""
    try:
        gas_num_largo = GAS_CONFIG.get(gas_num, {}).get("nombre_largo", gas_num)
        gas_den_largo = GAS_CONFIG.get(gas_den, {}).get("nombre_largo", gas_den)

        gdf = gpd.read_file(ruta_geojson).dissolve()
        with rasterio.open(ruta_num) as src_num, rasterio.open(ruta_den) as src_den:
            if gdf.crs != src_num.crs: gdf = gdf.to_crs(src_num.crs)
            
            # Enmascarar con la geometría de la región
            data_n, _ = mask(dataset=src_num, shapes=gdf.geometry, crop=True, nodata=np.nan)
            data_d, _ = mask(dataset=src_den, shapes=gdf.geometry, crop=True, nodata=np.nan)
            
            val_n = data_n[0].flatten()
            val_d = data_d[0].flatten()
            
            # Extraer solo píxeles válidos para ambos componentes concurrentemente
            valid_mask = ~np.isnan(val_n) & ~np.isnan(val_d) & (val_n > 0) & (val_d > 0)
            val_n = val_n[valid_mask]
            val_d = val_d[valid_mask]
            
            if val_n.size < 2:
                print("❌ No hay suficientes datos concurrentes para correlación.")
                return None
                
            corr_p, _ = pearsonr(val_n, val_d)
            corr_s, _ = spearmanr(val_n, val_d)
            
            fig, ax = plt.subplots(figsize=(9, 7))
            
            # Grafico Hexbin para manejar gran volumen de datos (densidad espacial)
            hb = ax.hexbin(val_d, val_n, gridsize=50, cmap='plasma', mincnt=1, bins='log')
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('log$_{10}$(Densidad de Píxeles)', fontsize=12)
            
            # Recortar atípicos extremos ajustando dinámicamente el rango en ambos ejes
            p01_d, p99_d = np.percentile(val_d, [1, 99])
            p01_n, p99_n = np.percentile(val_n, [1, 99])
            
            # Añadir un margen visual del 5%
            margen_d = (p99_d - p01_d) * 0.05
            margen_n = (p99_n - p01_n) * 0.05
            
            xlim_min = max(0, p01_d - margen_d) # En concentraciones el mínimo lógico es ~0
            xlim_max = p99_d + margen_d
            ylim_min = max(0, p01_n - margen_n)
            ylim_max = p99_n + margen_n
            
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            
            # Línea de tendencia lineal
            m, b = np.polyfit(val_d, val_n, 1)
            x_trend = np.linspace(xlim_min, xlim_max, 100)
            ax.plot(x_trend, m*x_trend + b, color='lime', linestyle='--', linewidth=2, label=f'Tendencia Lineal')
            
            ax.set_title(f"Correlación Espacial: {gas_num} vs {gas_den}\n{region_nombre.replace('_', ' ').title()} • {mes_str.capitalize()} {año}", fontsize=15, pad=15)
            ax.set_xlabel(f"Eje X: [{gas_den}] {gas_den_largo} (mol/m²)", fontsize=13)
            ax.set_ylabel(f"Eje Y: [{gas_num}] {gas_num_largo} (mol/m²)", fontsize=13)
            
            # Cuadro de métricas con identificación clara
            textstr = (
                f"Identificación de Gases:\n"
                f"$\\bullet$ Eje Y: {gas_num} ({gas_num_largo})\n"
                f"$\\bullet$ Eje X: {gas_den} ({gas_den_largo})\n"
                f"--------------------------------------\n"
                f"Pearson ($r$): {corr_p:.3f}\n"
                f"Spearman ($\\rho$): {corr_s:.3f}\n"
                f"Píxeles válidos: {val_n.size}"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
            ax.legend(loc='lower right')
            
            plt.tight_layout()
            
            # Guardar en la carpeta especificada del periodo
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"Correlacion_{gas_num}_vs_{gas_den}_{region_nombre}_{año}_{mes_str}.png"
            
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"📈 Gráfico de correlación guardado en: {output_path}")
            
            if return_fig: return fig
            else: plt.close(fig); return None
            
    except Exception as e:
        print(f"❌ Error al generar gráfica de correlación: {e}")
        return None

# --- GRAFICADOR SERIES TEMPORALES ---
def graficar_serie_temporal(csv_paths, region_nombre, gas_nombre, title_suffix, output_dir, is_ppb=False, return_fig=False):
    """Grafica la evolución con un área sombreada del rango de dispersión (P5 a P95) y la mediana como línea continua"""
    valid_dfs = [pd.read_csv(cp) for cp in csv_paths if cp and Path(cp).exists()]
    if not valid_dfs: return None
    
    df_all = pd.concat(valid_dfs, ignore_index=True)
    df_all['Fecha'] = pd.to_datetime(df_all['Fecha'])
    df_all = df_all.sort_values('Fecha')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sombreado de área entre P5 y P95
    ax.fill_between(df_all['Fecha'], df_all['P5'], df_all['P95'], color='dodgerblue', alpha=0.3, label='Rango de Dispersión (P5 - P95)')
    
    # Línea continua para la Mediana (P50)
    ax.plot(df_all['Fecha'], df_all['P50'], marker='o', linestyle='-', color='dodgerblue', markersize=5, linewidth=2, label='Mediana Espacial (P50)')
    
    if is_ppb:
        ax.set_title(f"Evolución Temporal (Superficie PPB) - {gas_nombre}\n{region_nombre.replace('_', ' ').title()} • {title_suffix}", fontsize=14)
        ax.set_ylabel(f"Concentración Superficial {gas_nombre} (ppb)", fontsize=12)
        out_path = output_dir / f"Serie_Temporal_Rango_PPB_{gas_nombre}_{title_suffix.replace(' ', '_')}.png"
    else:
        ax.set_title(f"Evolución Temporal (Columna Total) - {gas_nombre}\n{region_nombre.replace('_', ' ').title()} • {title_suffix}", fontsize=14)
        ax.set_ylabel(f"Concentración Columna {gas_nombre} (mol/m²)", fontsize=12)
        out_path = output_dir / f"Serie_Temporal_Rango_Columna_{gas_nombre}_{title_suffix.replace(' ', '_')}.png"

    ax.set_xlabel("Fecha", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300)
    print(f"📈 Serie temporal guardada en: {out_path}")
    
    if return_fig: return fig
    else: plt.close(fig); return None

def graficar_serie_temporal_dual(csv_paths_num, csv_paths_den, region_nombre, gas_num, gas_den, title_suffix, output_dir, return_fig=False):
    """Grafica en doble eje la evolución de dos gases durante un análisis de Ratio Maps usando Mediana y Sombreado"""
    valid_dfs_num = [pd.read_csv(cp) for cp in csv_paths_num if cp and Path(cp).exists()]
    valid_dfs_den = [pd.read_csv(cp) for cp in csv_paths_den if cp and Path(cp).exists()]
    
    if not valid_dfs_num or not valid_dfs_den: return None
    
    df_num = pd.concat(valid_dfs_num, ignore_index=True)
    df_den = pd.concat(valid_dfs_den, ignore_index=True)
    
    df_num['Fecha'] = pd.to_datetime(df_num['Fecha'])
    df_den['Fecha'] = pd.to_datetime(df_den['Fecha'])
    df_num = df_num.sort_values('Fecha')
    df_den = df_den.sort_values('Fecha')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Eje y principal (Gas Numerador)
    color1 = 'tab:blue'
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel(f'{gas_num} (mol/m²)', color=color1, fontsize=12)
    ax1.fill_between(df_num['Fecha'], df_num['P5'], df_num['P95'], color=color1, alpha=0.2)
    line1 = ax1.plot(df_num['Fecha'], df_num['P50'], color=color1, marker='o', linestyle='-', markersize=4, linewidth=2, label=f'{gas_num} (Mediana y Rango P5-P95)')
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.xticks(rotation=45)
    
    # Eje y secundario (Gas Denominador)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'{gas_den} (mol/m²)', color=color2, fontsize=12)
    ax2.fill_between(df_den['Fecha'], df_den['P5'], df_den['P95'], color=color2, alpha=0.2)
    line2 = ax2.plot(df_den['Fecha'], df_den['P50'], color=color2, marker='s', linestyle='-', markersize=4, linewidth=2, label=f'{gas_den} (Mediana y Rango P5-P95)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Unir leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    fig.suptitle(f"Evolución Temporal Comparativa (Mediana Espacial y Rango de Dispersión)\n{region_nombre.replace('_', ' ').title()} • {title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2) # Espacio extra para la leyenda inferior
    
    out_path = output_dir / f"Serie_Temporal_Rango_Dual_{gas_num}_vs_{gas_den}_{title_suffix.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=300)
    print(f"📈 Serie temporal comparativa guardada en: {out_path}")
    
    if return_fig: return fig
    else: plt.close(fig); return None

def regrid_geotiff(input_tiff_path, grid_resolution=100):
    carpeta = os.path.dirname(input_tiff_path)
    output_tiff_path = os.path.join(carpeta, "response_regrid.tiff")
    try:
        with rasterio.open(input_tiff_path) as ds:
            band1 = ds.read(1)
            transform, crs, nodata = ds.transform, ds.crs, ds.nodata
            if nodata is not None:
                if not np.issubdtype(band1.dtype, np.floating): band1 = band1.astype(np.float32)
                band1[band1 == nodata] = np.nan
            else:
                if not np.issubdtype(band1.dtype, np.floating): band1 = band1.astype(np.float32)

            coords, vals = [], []
            filas, columnas = band1.shape
            for fila in range(filas):
                for col in range(columnas):
                    val = band1[fila, col]
                    if not np.isnan(val):
                        lon, lat = transform * (col + 0.5, fila + 0.5)
                        coords.append((lon, lat))
                        vals.append(val)

        if not vals: return None
        df = pd.DataFrame(coords, columns=["lon", "lat"])
        df["value"] = vals
        lon_grid = np.linspace(df["lon"].min(), df["lon"].max(), grid_resolution)
        lat_grid = np.linspace(df["lat"].max(), df["lat"].min(), grid_resolution)
        
        ok = OrdinaryKriging(df["lon"].values, df["lat"].values, df["value"].values, variogram_model="spherical", verbose=False, enable_plotting=False)
        interpolado, _ = ok.execute("grid", lon_grid, lat_grid)
        
        if grid_resolution > 1:
             res_x = (lon_grid.max() - lon_grid.min()) / (len(lon_grid) - 1)
             res_y = (lat_grid.min() - lat_grid.max()) / (len(lat_grid) - 1) 
        else: res_x = res_y = 0

        left_edge = lon_grid.min() - res_x / 2; top_edge  = lat_grid.max() - res_y / 2 
        nuevo_transform = from_origin(left_edge, top_edge, res_x, abs(res_y))

        with rasterio.open(output_tiff_path, "w", driver="GTiff", height=interpolado.shape[0], width=interpolado.shape[1], count=1, dtype=interpolado.dtype, crs=crs, transform=nuevo_transform) as dst:
            dst.write(interpolado, 1)
        return output_tiff_path
    except Exception: return None

def generar_mapa_con_leyenda(ruta_tiff, ruta_geojson, title_date, year, cmap='viridis', alpha=0.75, producto="Data", unidad="Unit", return_fig=False):
    try:
        gdf = gpd.read_file(ruta_geojson).dissolve()
        region_nombre = Path(ruta_geojson).stem.replace("_", " ").title()
        with rasterio.open(ruta_tiff) as src:
            if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
            try: data, out_transform = mask(dataset=src, shapes=gdf.geometry, crop=True, nodata=np.nan)
            except ValueError: return None
            data = data[0]
            if np.all(np.isnan(data)): return None
            bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], out_transform)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]] 

        output_path = Path(ruta_tiff).with_suffix('.png')
        procesamiento = "Re-escalado (Kriging)" if 'regrid' in Path(ruta_tiff).stem.lower() else ("Calculado" if 'concentracion' in Path(ruta_tiff).stem.lower() else "Crudo")

        fig, ax = plt.subplots(figsize=(12, 10))
        cmap_obj = cmc.batlow if tiene_cmcrameri and cmap == 'batlow' else plt.get_cmap(cmap)
        
        # --- NUEVA LÓGICA DE MAPA PARA ESCALA DINÁMICA ---
        p02 = np.nanpercentile(data, 2)
        p95 = np.nanpercentile(data, 95)
        p98 = np.nanpercentile(data, 98)

        if "Relación" in producto or "Ratio" in producto:
            vmin = max(0, np.nanmin(data)) # Los ratios arrancan desde 0 generalmente
            if "HCHO" in producto and "NO2" in producto:
                # Topamos duramente en 5 o menos para no desdibujar la sensibilidad (0 a ~2 es el umbral de interés)
                vmax_real = min(5.0, max(2.0, p98))
            else:
                # Para otros ratios genéricos, el P95 es más seguro para evitar valores disparados al infinito
                vmax_real = p95
        else:
            vmin = p02
            vmax_real = p98
            
        norm = Normalize(vmin=vmin, vmax=vmax_real)
        
        img = ax.imshow(data, extent=extent, cmap=cmap_obj, norm=norm, alpha=alpha, zorder=10)
        gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, linestyle='--', zorder=11)
        
        # Agregar leyenda explicativa de regímenes si es HCHO/NO2
        if "HCHO" in producto and "NO2" in producto:
            props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
            regimes_text = (
                "Índice de Sensibilidad Fotoquímica\n"
                "HCHO (Formaldehído) / NO2 (Dióxido de Nitrógeno)\n"
                "--------------------------------------------------\n"
                "$\\bullet$ Ratio < 1 : Régimen limitado por COV\n"
                "$\\bullet$ Ratio 1 - 2 : Zona de Transición\n"
                "$\\bullet$ Ratio > 2 : Régimen limitado por NOx"
            )
            ax.text(0.02, 0.02, regimes_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=props, zorder=15)

        try: cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
        except Exception: pass

        ax.set_title(f"{producto} en {region_nombre} • {title_date.capitalize()} {year}\nDatos {procesamiento}", fontsize=16, pad=15)
        ax.set_xlabel("Longitud", fontsize=14); ax.set_ylabel("Latitud", fontsize=14); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.2)
        cbar = plt.colorbar(img, cax=cax); cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=20, fontsize=14)
        plt.tight_layout(); plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"🖼️ Mapa guardado en: {output_path}")
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception as e: print(f"❌ Error al generar mapa: {e}"); return None

def generar_mapa_comparativo(file_paths, aoi_path, producto, unidad, cmap, title_suffix, output_dir=None, return_fig=False):
    if not file_paths: return None
    try:
        gdf = gpd.read_file(aoi_path).dissolve()
        region_nombre = Path(aoi_path).stem.replace("_", " ").title()
        vmin, vmax = np.inf, -np.inf
        all_data, valid_extents, target_crs = {}, [], None 
        
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                if target_crs is None: target_crs = src.crs
                gdf_proj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
                try:
                    data, out_transform = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    data = data[0]
                    if np.all(np.isnan(data)):
                        all_data[file_path] = {'data': None, 'extent': None}; continue 
                    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], out_transform)
                    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
                    all_data[file_path] = {'data': data, 'extent': extent}
                    valid_extents.append(extent)
                except ValueError: all_data[file_path] = {'data': None, 'extent': None}

        if not valid_extents: return None
            
        # --- CÁLCULO ROBUSTO DE LÍMITES GLOBALES ---
        valid_arrays = [v['data'][~np.isnan(v['data'])] for v in all_data.values() if v['data'] is not None]
        if valid_arrays:
            merged_vals = np.concatenate(valid_arrays)
            p02 = np.percentile(merged_vals, 2)
            p95 = np.percentile(merged_vals, 95)
            p98 = np.percentile(merged_vals, 98)
            
            if "Ratio" in producto or "Relación" in producto:
                global_vmin = max(0, np.min(merged_vals))
                if "HCHO" in producto and "NO2" in producto:
                    global_vmax = min(5.0, max(2.0, p98))
                else:
                    global_vmax = p95
            else:
                global_vmin = p02
                global_vmax = p98
        else:
            global_vmin, global_vmax = 0, 1

        global_left, global_right = min(e[0] for e in valid_extents), max(e[1] for e in valid_extents)
        global_bottom, global_top = min(e[2] for e in valid_extents), max(e[3] for e in valid_extents)
        gdf_plot = gdf.to_crs(target_crs) if target_crs and gdf.crs != target_crs else gdf
            
        n = len(file_paths)
        if n <= 4: nrows, ncols = 1, n
        elif n <= 8: nrows, ncols = 2, (n + 1) // 2
        elif n <= 12: nrows, ncols = 3, 4
        else: nrows, ncols = (n + 3) // 4, 4 
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 7)) 
        axs = np.atleast_1d(axs).flatten() 
        cmap_obj = cmc.batlow if tiene_cmcrameri and cmap == 'batlow' else plt.get_cmap(cmap)
        
        norm = Normalize(vmin=global_vmin, vmax=global_vmax)
        img = None 
        
        for i, file_path in enumerate(file_paths):
            ax = axs[i]; plot_data = all_data[file_path]
            title = next((m for m in meses_dict.values() if m.lower() in str(file_path).lower()), file_path.stem)
            try:
                parts = file_path.parent.name.split('_') 
                if len(parts) >= 2 and parts[-1].isdigit(): title = f"{parts[-2].capitalize()} {parts[-1]}"
            except Exception: pass
            
            if plot_data['data'] is not None:
                img = ax.imshow(plot_data['data'], extent=plot_data['extent'], cmap=cmap_obj, norm=norm, alpha=0.8, zorder=10)
                gdf_plot.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, linestyle='--', zorder=11)
                ax.set_title(title, fontsize=11) 
                try: cx.add_basemap(ax, crs=gdf_plot.crs, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
                except Exception: pass 
            else: ax.set_title(f"{title}\n(Sin datos)", fontsize=12); ax.set_facecolor('0.95') 
            
            ax.set_xlim(global_left, global_right); ax.set_ylim(global_bottom, global_top)
            ax.tick_params(labelbottom=False, labelleft=False)
            if i // ncols == nrows - 1: ax.tick_params(labelbottom=True); plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            if i % ncols == 0: ax.tick_params(labelleft=True); plt.setp(ax.get_yticklabels(), fontsize=10)

        for i in range(n, len(axs)): axs[i].set_visible(False)
        fig.suptitle(f"{producto} en {region_nombre} • {title_suffix}", fontsize=18) 
        
        if img: 
            fig.tight_layout(rect=[0.04, 0.05, 0.85, 0.95]); fig.canvas.draw() 
            cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
            cbar = fig.colorbar(img, cax=cbar_ax)
            cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=25, fontsize=14)
            fig.supxlabel('Longitud', fontsize=16, y=0.02); fig.supylabel('Latitud', fontsize=16, x=0.01)

        if output_dir is None:
            output_dir = file_paths[0].parent.parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"Mapa_Comparativo_{producto.split()[0]}_{title_suffix.replace(' ', '_')}.png"
        
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception as e: print(f"❌ Error mapa comparativo: {e}"); return None

def analyze_tiff_statistics(tiff_path: str, return_fig=False):
    try:
        with rasterio.open(tiff_path) as src:
            image_data = src.read(1).astype(np.float32)
            if src.nodata is not None: image_data[image_data == src.nodata] = np.nan
            pixel_values = image_data[~np.isnan(image_data)]
        
        if pixel_values.size == 0: return None
        mean_val, std_val, median_val = np.nanmean(pixel_values), np.nanstd(pixel_values), np.nanmedian(pixel_values) 

        stats_df = pd.DataFrame({'Statistic': ['Minimum', 'Maximum', 'Median', 'Mean', 'Std Dev', 'Data Points'], 'Value': [np.nanmin(pixel_values), np.nanmax(pixel_values), f"{median_val:.3e}", f"{mean_val:.3e}", f"{std_val:.3e}", pixel_values.size]})
        stats_df.to_csv(f"{Path(tiff_path).with_suffix('')}_statistics.csv", index=False)

        fig = plt.figure(figsize=(10, 6))
        
        # Ajuste de bins para evitar un histograma distorsionado por outliers masivos
        p01 = np.nanpercentile(pixel_values, 1)
        p99 = np.nanpercentile(pixel_values, 99)
        clipped_values = pixel_values[(pixel_values >= p01) & (pixel_values <= p99)]
        
        plt.hist(clipped_values, bins=50, density=True, alpha=0.7, color='skyblue', label='Distribución de Datos (P1 a P99)')
        x_norm = np.linspace(clipped_values.min(), clipped_values.max(), 100)
        pdf_fitted = norm.pdf(x_norm, mean_val, std_val)
        plt.plot(x_norm, pdf_fitted, 'r-', linewidth=2, label=f'Distribución Normal (μ={mean_val:.2e}, σ={std_val:.2e})')
        plt.title(f"Distribución de Valores para {Path(tiff_path).stem}")
        plt.xlabel("Valor del Píxel"); plt.ylabel("Densidad"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{Path(tiff_path).with_suffix('')}_distribution.png")
        
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception: return None

# --- FUNCIONES DE EJECUCIÓN ---

def run_processing(params, cancel_event):
    matplotlib.use('Agg')
    gas_info = params['gas_info']
    gas_name = gas_info['nombre_corto']
    aoi_path, region_nombre = params['aoi_path'], Path(params['aoi_path']).stem
    time_start, time_end = params['start_date'], params['end_date']
    
    ruta_blh_para_calculo = ruta_gas_para_procesar = ruta_blh_para_analisis = ruta_concentracion_final = ruta_csv_gas = ruta_csv_ppb = None

    print(f"\n--- 🗓️  PROCESANDO {gas_name} en {region_nombre.replace('_', ' ').title()} para {time_start} a {time_end} ---")
    
    res_gas = descargar_y_promediar_gas(time_start, time_end, aoi_path, params['output_name'], gas_info, cancel_event, generar_serie=params.get('generar_serie', False))
    if not res_gas or not res_gas[0] or cancel_event.is_set():
        if res_gas and res_gas[2]: shutil.rmtree(res_gas[2], ignore_errors=True)
        return None, None, None, None, None
        
    ruta_crudo_gas, ruta_csv_gas, temp_dir, daily_tiffs, daily_dates = res_gas

    if ruta_crudo_gas and ruta_crudo_gas.exists():
        ruta_gas_para_procesar = ruta_crudo_gas 
        if params['do_regrid']:
            if (ruta_regrid_str := regrid_geotiff(str(ruta_crudo_gas))): ruta_gas_para_procesar = Path(ruta_regrid_str).resolve()
        
        if not params.get('do_comparative_map', False):
            if params['estadisticas'] and (fig := analyze_tiff_statistics(str(ruta_gas_para_procesar), return_fig=params['show_plots'])): params['fig_queue'].put(fig)
            if params['generar_mapas'] and (fig := generar_mapa_con_leyenda(ruta_gas_para_procesar, aoi_path, params['title_date'], params['year'], cmap=params['palette_gas'], producto=f"{gas_name} (Columna)", unidad="(mol/m²)", return_fig=params['show_plots'])): params['fig_queue'].put(fig)

    if cancel_event.is_set(): 
        if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, ruta_csv_gas, None

    if params.get('transform_method') == 'petetin':
        print(f"\n🌍 Descargando datos de BLH (H) para: {params['title_date']} {params['year']}")
        bounding_box = geojson_to_coords(aoi_path)
        if bounding_box:
            blh_output_dir = BASE_OUTPUT_PATH / f"BLH/{params['year']}/{region_nombre}/{params['output_name_blh']}"
            blh_output_dir.mkdir(parents=True, exist_ok=True)
            nombre_base_blh = f"BLH_ERA5_{region_nombre}_{params['year']}_{params['title_date']}"
            netcdf_file, tiff_file_blh = blh_output_dir / f"{nombre_base_blh}.nc", blh_output_dir / f"{nombre_base_blh}.tiff"
            
            exito_descarga = False
            era5_area_coords = [bounding_box[3], bounding_box[0], bounding_box[1], bounding_box[2]]
            if params.get('choice_mode') == 'dia': exito_descarga = descargar_blh_era5_diario(datetime.strptime(params['start_date'], '%Y-%m-%d'), datetime.strptime(params['start_date'], '%Y-%m-%d'), era5_area_coords, netcdf_file)
            else:
                 if 'month' in params: exito_descarga = descargar_blh_era5(params['year'], params['month'], era5_area_coords, netcdf_file)

            if exito_descarga:
                netcdf_file_tagged = blh_output_dir / f"{nombre_base_blh}_tagged.nc"
                if params['formato_salida'] == "GeoTIFF":
                    if convertir_nc_a_tiff(netcdf_file, tiff_file_blh): ruta_blh_para_calculo = ruta_blh_para_analisis = tiff_file_blh
                else: 
                    if tag_nc_with_crs(netcdf_file, netcdf_file_tagged): ruta_blh_para_calculo = ruta_blh_para_analisis = netcdf_file_tagged
                
                if ruta_blh_para_analisis and not params.get('do_comparative_map', False):
                    if params['estadisticas'] and (fig := analyze_tiff_statistics(str(ruta_blh_para_analisis), return_fig=params['show_plots'])): params['fig_queue'].put(fig)
                    if params['generar_mapas'] and (fig := generar_mapa_con_leyenda(ruta_blh_para_analisis, aoi_path, params['title_date'], params['year'], cmap=params['palette_blh'], producto="Capa Límite (BLH)", unidad="(m)", return_fig=params['show_plots'])): params['fig_queue'].put(fig)
    
    if cancel_event.is_set(): 
        if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)
        return ruta_gas_para_procesar, None, None, ruta_csv_gas, None

    can_calculate = (ruta_blh_para_calculo is not None) if params.get('transform_method') == 'petetin' else bool(ruta_gas_para_procesar and params.get('transform_method'))
    if can_calculate:
        folder_leaf = "Savenets Equation" if params['transform_method'] == 'savanets' else (f"{params.get('transform_value', 0)} transformed" if params['transform_method'] == 'custom' else params.get('output_name_blh', params['output_name']).replace('BLH', 'Concentracion').replace(f'Datos_{gas_name}', 'Concentracion'))
        calc_output_dir = BASE_OUTPUT_PATH / f"Calculos_{gas_name}/{params['year']}/{region_nombre}/{folder_leaf}"
        calc_output_dir.mkdir(parents=True, exist_ok=True)

        ruta_concentracion_final = procesar_concentracion_gas(
            ruta_gas_para_procesar, ruta_blh_para_calculo, calc_output_dir, region_nombre, 
            params['year'], params['title_date'], gas_info, params['formato_salida'],
            metodo_transform=params['transform_method'], valor_custom=params.get('transform_value'), suffix="_regrid" if params['do_regrid'] and ruta_gas_para_procesar.name != "response.tiff" else ""
        )
        
        # --- EXTRAER SERIE TEMPORAL DE PPB SI FUE SOLICITADO ---
        if params.get('generar_serie') and daily_tiffs:
            csv_ppb_name = calc_output_dir / f"Serie_Temporal_Rango_PPB_{gas_name}_{params['title_date']}.csv"
            ruta_csv_ppb = extraer_serie_ppb(daily_tiffs, daily_dates, ruta_blh_para_calculo, params['transform_method'], params.get('transform_value'), gas_info, aoi_path, csv_ppb_name)

        if ruta_concentracion_final and not params.get('do_comparative_map', False):
            if params['estadisticas'] and (fig := analyze_tiff_statistics(str(ruta_concentracion_final), return_fig=params['show_plots'])): params['fig_queue'].put(fig)
            if params['generar_mapas'] and (fig := generar_mapa_con_leyenda(ruta_concentracion_final, aoi_path, params['title_date'], params['year'], cmap=params['palette_transform'], producto=f"Concentración de {gas_name}", unidad="(ppb)", return_fig=params['show_plots'])): params['fig_queue'].put(fig)

    # Limpiar siempre el directorio temporal luego del proceso finalizado
    if temp_dir and temp_dir.exists():
        try: shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception: pass

    if params['do_zip'] and ruta_crudo_gas: comprimir_directorio(ruta_crudo_gas.parent, ruta_crudo_gas.parent.parent / f"{ruta_crudo_gas.parent.name}_{gas_name}-crudo.zip")

    return ruta_gas_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final, ruta_csv_gas, ruta_csv_ppb

def run_processing_ratio(params, cancel_event):
    """Ejecuta el pipeline especializado para calcular el ratio espacial entre dos gases."""
    matplotlib.use('Agg')
    gas_num_info, gas_den_info = params['gas_num_info'], params['gas_den_info']
    aoi_path, region_nombre = params['aoi_path'], Path(params['aoi_path']).stem
    
    print(f"\n==========================================================")
    print(f"🔬 ANÁLISIS MULTIGAS: RELACIÓN {gas_num_info['nombre_corto']} / {gas_den_info['nombre_corto']}")
    print(f"   Analizando sensibilidad fotoquímica local (COV vs NOx)...")
    print(f"==========================================================")
    
    # 1. Descargar Gas Numerador (Ej: HCHO)
    params_num = params.copy()
    params_num['gas_info'] = gas_num_info
    params_num['output_name'] = params['output_name'].replace(params.get('gas_info', gas_num_info)['nombre_corto'], gas_num_info['nombre_corto'])
    res_num = descargar_y_promediar_gas(params['start_date'], params['end_date'], aoi_path, params_num['output_name'], gas_num_info, cancel_event, generar_serie=params.get('generar_serie', False))
    if not res_num or not res_num[0] or cancel_event.is_set():
        if res_num and res_num[2]: shutil.rmtree(res_num[2], ignore_errors=True)
        return None, None, None
    ruta_num, csv_num, temp_num, _, _ = res_num

    # 2. Descargar Gas Denominador (Ej: NO2)
    params_den = params.copy()
    params_den['gas_info'] = gas_den_info
    params_den['output_name'] = params['output_name'].replace(params.get('gas_info', gas_num_info)['nombre_corto'], gas_den_info['nombre_corto'])
    res_den = descargar_y_promediar_gas(params['start_date'], params['end_date'], aoi_path, params_den['output_name'], gas_den_info, cancel_event, generar_serie=params.get('generar_serie', False))
    if not res_den or not res_den[0] or cancel_event.is_set(): 
        if temp_num: shutil.rmtree(temp_num, ignore_errors=True)
        if res_den and res_den[2]: shutil.rmtree(res_den[2], ignore_errors=True)
        return None, None, None
    ruta_den, csv_den, temp_den, _, _ = res_den

    # 3. Calcular el Ratio (HCHO/NO2)
    gas_num_str = gas_num_info['nombre_corto']
    gas_den_str = gas_den_info['nombre_corto']
    periodo = params.get('folder_period', 'Periodo')
    output_dir = BASE_OUTPUT_PATH / f"Ratios/{gas_num_str}_{gas_den_str}/{params['year']}/{region_nombre}/{periodo}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ruta_ratio = calcular_y_guardar_ratio(ruta_num, ruta_den, output_dir, gas_num_info['nombre_corto'], gas_den_info['nombre_corto'], region_nombre, params['year'], params['title_date'])

    # 4. Generar Estadísticas, Matriz de Correlación y Mapas
    if ruta_ratio and not params.get('do_comparative_map', False):
        if params['estadisticas']:
            if (fig := analyze_tiff_statistics(str(ruta_ratio), return_fig=params['show_plots'])): params['fig_queue'].put(fig)
            
            # --- Matriz / Gráfico de Correlación ---
            if (fig_corr := calcular_y_graficar_correlacion(ruta_num, ruta_den, aoi_path, gas_num_info['nombre_corto'], gas_den_info['nombre_corto'], region_nombre, params['year'], params['title_date'], output_dir, return_fig=params['show_plots'])):
                params['fig_queue'].put(fig_corr)

        if params['generar_mapas']:
            # Sugerimos una escala divergente si es la default (Spectral ayuda mucho a la transición HCHO/NO2)
            cmap = "Spectral_r" if params['palette_gas'] in ['viridis', 'plasma', 'inferno'] else params['palette_gas']
            fig = generar_mapa_con_leyenda(ruta_ratio, aoi_path, params['title_date'], params['year'], cmap=cmap, producto=f"Relación {gas_num_info['nombre_corto']}/{gas_den_info['nombre_corto']}", unidad="(adimensional)", return_fig=params['show_plots'])
            if fig: params['fig_queue'].put(fig)

    if temp_num and Path(temp_num).exists(): shutil.rmtree(temp_num, ignore_errors=True)
    if temp_den and Path(temp_den).exists(): shutil.rmtree(temp_den, ignore_errors=True)

    return ruta_ratio, csv_num, csv_den

# ==============================================================================
# SECCIÓN 2: CLASE DE LA APLICACIÓN TKINTER
# ==============================================================================
class GeoApp:
    def __init__(self, root, available_regions):
        self.root = root
        self.root.title("Procesador de Datos Atmosféricos Multi-Gas")
        self.root.geometry("850x950") 
        self.fig_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.available_regions = available_regions
        self.cancel_event = threading.Event()
        
        self.ancho_spinbox_ano = 6
        self.ancho_combobox_mes = 12

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_gas_widgets(main_frame)
        self.create_date_widgets(main_frame)
        self.create_region_widgets(main_frame)
        self.create_options_widgets(main_frame)
        self.create_console_widgets(main_frame)
        self.create_action_buttons(main_frame)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, self.check_fig_queue)
        
        self.toggle_date_widgets()
        self.on_gas_tab_changed(None)

    def on_closing(self):
        if self.cancel_event: self.cancel_event.set()
        sys.stdout = self.original_stdout
        self.root.destroy()
        
    def create_gas_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="0. Selección de Gas / Relación (Ratio Maps)", padding="10")
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.gas_notebook = ttk.Notebook(frame)
        self.gas_notebook.pack(fill=tk.X, expand=True)
        
        self.tab_sub_blh = ttk.Frame(self.gas_notebook, padding="10")
        self.tab_post_blh = ttk.Frame(self.gas_notebook, padding="10")
        self.tab_ratio = ttk.Frame(self.gas_notebook, padding="10") 
        
        self.gas_notebook.add(self.tab_sub_blh, text="Gases Sub-BLH (Transformables)")
        self.gas_notebook.add(self.tab_post_blh, text="Gases Post-BLH (Solo Columna)")
        self.gas_notebook.add(self.tab_ratio, text="Mapas de Relación (Ej: HCHO/NO2)")
        
        ttk.Label(self.tab_sub_blh, text="Gas (Confinado en PBL):").pack(side=tk.LEFT, padx=5)
        self.combo_sub_blh = ttk.Combobox(self.tab_sub_blh, values=list(GASES_SUB_BLH.keys()), state="readonly")
        self.combo_sub_blh.pack(side=tk.LEFT, padx=5)
        if GASES_SUB_BLH: self.combo_sub_blh.set(list(GASES_SUB_BLH.keys())[0])
        
        ttk.Label(self.tab_post_blh, text="Gas (Troposfera Extendida):").pack(side=tk.LEFT, padx=5)
        self.combo_post_blh = ttk.Combobox(self.tab_post_blh, values=list(GASES_POST_BLH.keys()), state="readonly")
        self.combo_post_blh.pack(side=tk.LEFT, padx=5)
        if GASES_POST_BLH: self.combo_post_blh.set(list(GASES_POST_BLH.keys())[0])

        # --- Interfaz de Ratio Maps ---
        ttk.Label(self.tab_ratio, text="Numerador:").pack(side=tk.LEFT, padx=5)
        self.combo_ratio_num = ttk.Combobox(self.tab_ratio, values=list(GAS_CONFIG.keys()), state="readonly", width=8)
        self.combo_ratio_num.pack(side=tk.LEFT, padx=5)
        self.combo_ratio_num.set("HCHO")

        ttk.Label(self.tab_ratio, text=" / Denominador:").pack(side=tk.LEFT, padx=5)
        self.combo_ratio_den = ttk.Combobox(self.tab_ratio, values=list(GAS_CONFIG.keys()), state="readonly", width=8)
        self.combo_ratio_den.pack(side=tk.LEFT, padx=5)
        self.combo_ratio_den.set("NO2")

        self.gas_notebook.bind("<<NotebookTabChanged>>", self.on_gas_tab_changed)
        
    def on_gas_tab_changed(self, event):
        current_tab = self.gas_notebook.index("current")
        if current_tab == 1 or current_tab == 2: # Post BLH o Ratio
            if hasattr(self, 'do_transform'):
                self.do_transform.set(False)
                self.transform_checkbutton.config(state="disabled")
                self.transform_method_combo.config(state="disabled")
        else:
            if hasattr(self, 'do_transform'):
                self.update_options_state() 

    def create_date_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="1. Selección de Fecha", padding="10")
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.date_choice = tk.StringVar(value="mes")
        radio_frame = ttk.Frame(frame); radio_frame.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(radio_frame, text="Mes Particular", variable=self.date_choice, value="mes", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Año Completo", variable=self.date_choice, value="anio", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Día Puntual", variable=self.date_choice, value="dia", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Rango Días", variable=self.date_choice, value="rango", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(radio_frame, text="Rango Meses", variable=self.date_choice, value="rango_meses", command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)
        
        self.date_widgets_frame = ttk.Frame(frame); self.date_widgets_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)
        
        self.mes_label, self.mes_combo = ttk.Label(self.date_widgets_frame, text="Mes:"), ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes); self.mes_combo.set("Enero")
        self.ano_label, self.ano_spin = ttk.Label(self.date_widgets_frame, text="Año:"), tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.ano_label_completo, self.ano_spin_completo = ttk.Label(self.date_widgets_frame, text="Año:"), tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.dia_label, self.dia_cal = ttk.Label(self.date_widgets_frame, text="Fecha:"), DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_label1, self.rango_cal1 = ttk.Label(self.date_widgets_frame, text="Desde:"), DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_label2, self.rango_cal2 = ttk.Label(self.date_widgets_frame, text="Hasta:"), DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_mes_label_ini, self.rango_mes_combo_ini, self.rango_ano_spin_ini = ttk.Label(self.date_widgets_frame, text="Desde:"), ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes), tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano); self.rango_mes_combo_ini.set("Enero")
        self.rango_mes_label_fin, self.rango_mes_combo_fin, self.rango_ano_spin_fin = ttk.Label(self.date_widgets_frame, text="Hasta:"), ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes), tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano); self.rango_mes_combo_fin.set("Marzo")

    def toggle_date_widgets(self):
        for widget in self.date_widgets_frame.winfo_children(): widget.grid_forget()
        choice = self.date_choice.get()
        if choice == "mes":
            self.mes_label.grid(row=0, column=0, sticky="w", padx=5); self.mes_combo.grid(row=0, column=1, sticky="w", padx=5)
            self.ano_label.grid(row=0, column=2, sticky="w", padx=5); self.ano_spin.grid(row=0, column=3, sticky="w", padx=5)
        elif choice == "anio": self.ano_label_completo.grid(row=0, column=0, sticky="w", padx=5); self.ano_spin_completo.grid(row=0, column=1, sticky="w", padx=5)
        elif choice == "dia": self.dia_label.grid(row=0, column=0, sticky="w", padx=5); self.dia_cal.grid(row=0, column=1, sticky="w", padx=5)
        elif choice == "rango":
            self.rango_label1.grid(row=0, column=0, sticky="w", padx=5); self.rango_cal1.grid(row=0, column=1, sticky="w", padx=5)
            self.rango_label2.grid(row=0, column=2, sticky="w", padx=5); self.rango_cal2.grid(row=0, column=3, sticky="w", padx=5)
        elif choice == "rango_meses":
            self.rango_mes_label_ini.grid(row=0, column=0, sticky="w", padx=5); self.rango_mes_combo_ini.grid(row=0, column=1, sticky="w", padx=5); self.rango_ano_spin_ini.grid(row=0, column=2, sticky="w", padx=5)
            self.rango_mes_label_fin.grid(row=1, column=0, sticky="w", padx=5, pady=(5,0)); self.rango_mes_combo_fin.grid(row=1, column=1, sticky="w", padx=5, pady=(5,0)); self.rango_ano_spin_fin.grid(row=1, column=2, sticky="w", padx=5, pady=(5,0))
        self.update_options_state()

    def create_region_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Selección de Región", padding="10")
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.region_mode = tk.StringVar(value="list")
        rb_frame = ttk.Frame(frame); rb_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(rb_frame, text="Seleccionar de Lista Precargada", variable=self.region_mode, value="list", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(rb_frame, text="Coordenadas Manuales (BBox)", variable=self.region_mode, value="manual", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        
        self.region_container = ttk.Frame(frame); self.region_container.pack(fill=tk.X, expand=True, pady=5)
        self.frame_list = ttk.Frame(self.region_container)
        ttk.Label(self.frame_list, text="Región:").pack(side=tk.LEFT, padx=5)
        self.region_combo = ttk.Combobox(self.frame_list, values=self.available_regions, state="readonly", width=40); self.region_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.available_regions:
            if "region_metropolitana_de_santiago" in self.available_regions: self.region_combo.set("region_metropolitana_de_santiago")
            else: self.region_combo.set(self.available_regions[0])
            
        self.frame_manual = ttk.Frame(self.region_container)
        nm_frame = ttk.Frame(self.frame_manual); nm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(nm_frame, text="Nombre de Zona (sin espacios):").pack(side=tk.LEFT)
        self.manual_name_var = tk.StringVar(); ttk.Entry(nm_frame, textvariable=self.manual_name_var, width=30).pack(side=tk.LEFT, padx=5)
        
        coord_frame = ttk.Frame(self.frame_manual); coord_frame.pack(fill=tk.X, pady=5)
        self.min_lon, self.min_lat = tk.DoubleVar(value=-71.0), tk.DoubleVar(value=-33.6)
        self.max_lon, self.max_lat = tk.DoubleVar(value=-70.0), tk.DoubleVar(value=-33.0)
        
        ttk.Label(coord_frame, text="Min Longitud (W):").grid(row=0, column=0, padx=5, pady=2, sticky="e"); ttk.Entry(coord_frame, textvariable=self.min_lon, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(coord_frame, text="Max Longitud (E):").grid(row=0, column=2, padx=5, pady=2, sticky="e"); ttk.Entry(coord_frame, textvariable=self.max_lon, width=10).grid(row=0, column=3, padx=5, pady=2)
        ttk.Label(coord_frame, text="Min Latitud (S):").grid(row=1, column=0, padx=5, pady=2, sticky="e"); ttk.Entry(coord_frame, textvariable=self.min_lat, width=10).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(coord_frame, text="Max Latitud (N):").grid(row=1, column=2, padx=5, pady=2, sticky="e"); ttk.Entry(coord_frame, textvariable=self.max_lat, width=10).grid(row=1, column=3, padx=5, pady=2)
        self.toggle_region_mode()

    def toggle_region_mode(self):
        self.frame_list.pack_forget(); self.frame_manual.pack_forget()
        if self.region_mode.get() == "list": self.frame_list.pack(fill=tk.X, expand=True)
        else: self.frame_manual.pack(fill=tk.X, expand=True)

    def create_options_widgets(self, parent):
        proc_frame = ttk.LabelFrame(parent, text="3. Opciones de Processing", padding="10")
        proc_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.do_stats = tk.BooleanVar(value=True); self.do_maps = tk.BooleanVar(value=True)
        self.do_transform = tk.BooleanVar(value=True); self.do_zip = tk.BooleanVar()
        self.do_regrid = tk.BooleanVar(value=False); self.do_comparative_map = tk.BooleanVar(value=False)
        self.do_timeseries = tk.BooleanVar(value=True) 
        
        self.transform_checkbutton = ttk.Checkbutton(proc_frame, text="Transformar a superficie", variable=self.do_transform, command=self.toggle_transform_options)
        self.transform_checkbutton.pack(anchor="w")
        self.transform_options_frame = ttk.Frame(proc_frame); self.transform_options_frame.pack(anchor="w", padx=20)
        self.transform_method_var = tk.StringVar(value="Descargar BLH y Calcular Concentración (H. Petetin Mode)")
        self.transform_method_combo = ttk.Combobox(self.transform_options_frame, textvariable=self.transform_method_var, values=["Descargar BLH y Calcular Concentración (H. Petetin Mode)", "Ecuación de Savanets (10km de altitud)", "Valor escrito por el usuario en pantalla"], state="readonly", width=50)
        self.transform_method_combo.pack(anchor="w")
        
        ttk.Checkbutton(proc_frame, text="Re-escalar datos originales (Kriging)", variable=self.do_regrid).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Analizar Estadísticas (Distribución y Correlación)", variable=self.do_stats).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Generar Mapas Individuales", variable=self.do_maps).pack(anchor="w")
        self.comp_map_checkbutton = ttk.Checkbutton(proc_frame, text="Generar Mapa Comparativo (sólo Año/Rango Meses)", variable=self.do_comparative_map)
        self.comp_map_checkbutton.pack(anchor="w")
        
        self.timeseries_checkbutton = ttk.Checkbutton(proc_frame, text="Graficar Serie Temporal (Mediana y Rango P5-P95)", variable=self.do_timeseries)
        self.timeseries_checkbutton.pack(anchor="w")
        
        ttk.Checkbutton(proc_frame, text="Comprimir resultados originales al finalizar", variable=self.do_zip).pack(anchor="w")
        
        ttk.Separator(proc_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(proc_frame, text="Formato de Salida (Ráster):").pack(anchor='w', pady=(5,0))
        self.formato_salida_var = tk.StringVar(value="GeoTIFF")
        self.formato_combo = ttk.Combobox(proc_frame, textvariable=self.formato_salida_var, values=["GeoTIFF", "NetCDF4", "ASCII Grid (.asc)"], state="readonly"); self.formato_combo.pack(fill=tk.X, anchor='w')
        
        vis_frame = ttk.LabelFrame(parent, text="4. Visualización", padding="10")
        vis_frame.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
        self.show_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Mostrar gráficos/mapas generados", variable=self.show_plots).pack(anchor="w")
        ttk.Separator(vis_frame, orient='horizontal').pack(fill='x', pady=10)
        
        paleta_opciones = list(paletas_colores.keys())
        ttk.Label(vis_frame, text="Color Gas Columna/Ratio:").pack(anchor='w')
        self.palette_gas_combo = ttk.Combobox(vis_frame, values=paleta_opciones, state="readonly"); self.palette_gas_combo.pack(fill=tk.X); self.palette_gas_combo.set("viridis")
        
        ttk.Label(vis_frame, text="Color Mapa BLH:").pack(anchor='w', pady=(5,0))
        self.palette_blh_combo = ttk.Combobox(vis_frame, values=paleta_opciones, state="readonly"); self.palette_blh_combo.pack(fill=tk.X); self.palette_blh_combo.set("turbo")

        ttk.Label(vis_frame, text="Color Gas Transformado:").pack(anchor='w', pady=(5,0))
        self.palette_transform_combo = ttk.Combobox(vis_frame, values=paleta_opciones, state="readonly"); self.palette_transform_combo.pack(fill=tk.X); self.palette_transform_combo.set("inferno")

        ttk.Separator(vis_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(vis_frame, text="Verificación Rápida de Nubosidad:", font=("Default", 9, "bold")).pack(anchor='w', pady=(0, 5))
        cloud_frame = ttk.Frame(vis_frame); cloud_frame.pack(fill=tk.X)
        self.btn_cloud = ttk.Button(cloud_frame, text="☁️ Calcular % Nubes", command=self.on_cloud_click); self.btn_cloud.pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_cloud_result = ttk.Label(cloud_frame, text="--- %", background="black", foreground="#00ff00", font=("Courier", 12, "bold"), padding=5, width=10, anchor="center"); self.lbl_cloud_result.pack(side=tk.LEFT)

    def on_cloud_click(self):
        try:
            params = self.get_params(); self.btn_cloud.config(state="disabled"); self.lbl_cloud_result.config(text="Calc...", foreground="yellow")
            if params: threading.Thread(target=self.run_cloud_check, args=(params,), daemon=True).start()
        except Exception as e: self.btn_cloud.config(state="normal"); self.lbl_cloud_result.config(text="Error", foreground="red")

    def run_cloud_check(self, params):
        try:
            t_start, t_end, route = params.get('start_date'), params.get('end_date'), params.get('aoi_path')
            if not t_start or not t_end or not route: self.update_cloud_ui("Datos Insuf.", "red"); return
            percent = calcular_estadisticas_nubosidad(t_start, t_end, route)
            if percent is not None:
                color = "#00ff00" if percent < 20 else ("orange" if percent < 50 else "red")
                self.update_cloud_ui(f"{percent:.1f}%", color)
            else: self.update_cloud_ui("Error", "red")
        except Exception: self.update_cloud_ui("Error", "red")
    
    def update_cloud_ui(self, text, color):
        self.root.after(0, lambda: [self.lbl_cloud_result.config(text=text, foreground=color), self.btn_cloud.config(state="normal")])
    
    def toggle_transform_options(self):
        if self.do_transform.get(): self.transform_method_combo.config(state="readonly")
        else: self.transform_method_combo.config(state="disabled")

    def update_options_state(self):
        choice = self.date_choice.get()
        current_tab = self.gas_notebook.index("current")
        
        if choice in ['mes', 'anio', 'rango_meses', 'dia', 'rango'] and current_tab == 0:
            self.transform_checkbutton.config(state="normal")
            if self.do_transform.get(): self.transform_method_combo.config(state="readonly")
        else:
            self.transform_checkbutton.config(state="disabled"); self.do_transform.set(False); self.transform_method_combo.config(state="disabled")

        if choice in ['anio', 'rango_meses']: self.comp_map_checkbutton.config(state="normal")
        else: self.comp_map_checkbutton.config(state="disabled"); self.do_comparative_map.set(False)

    def create_console_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Consola de Salida", padding="10")
        frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        parent.grid_rowconfigure(4, weight=1); parent.grid_columnconfigure(0, weight=1)
        self.console = scrolledtext.ScrolledText(frame, state='disabled', height=14, wrap=tk.WORD, bg="black", fg="white", font=("Courier New", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        sys.stdout = self.TextRedirector(self)

    def create_action_buttons(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(frame, text="Iniciar Proceso", command=self.start_processing_thread); self.run_button.pack(side=tk.RIGHT, padx=5)
        self.btn_download_all = ttk.Button(frame, text="Descargar todas las regiones", command=self.start_batch_processing_thread); self.btn_download_all.pack(side=tk.RIGHT, padx=5)
        self.cancel_button = ttk.Button(frame, text="Cancelar", command=self.request_cancellation, state="disabled"); self.cancel_button.pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame, text="Salir", command=self.on_closing).pack(side=tk.RIGHT)

    def start_processing_thread(self):
        self.cancel_event.clear(); self.run_button.config(state="disabled"); self.btn_download_all.config(state="disabled"); self.cancel_button.config(state="normal")
        self.clear_console()
        print("--- INICIANDO PROCESO (Región Única) ---")
        try:
            params = self.get_params()
            if params: threading.Thread(target=self.start_single_processing, args=(params, self.cancel_event), daemon=True).start()
            else: self.processing_finished()
        except Exception as e: messagebox.showerror("Error", str(e)); self.processing_finished()

    def start_batch_processing_thread(self):
        self.cancel_event.clear(); self.run_button.config(state="disabled"); self.btn_download_all.config(state="disabled"); self.cancel_button.config(state="normal")
        self.clear_console(); print("--- INICIANDO PROCESO POR LOTES ---")
        try:
            params = self.get_params(ignore_region=True)
            if params: threading.Thread(target=self.start_batch_processing, args=(params, self.cancel_event), daemon=True).start()
            else: self.processing_finished()
        except Exception as e: messagebox.showerror("Error", str(e)); self.processing_finished()

    def start_single_processing(self, params, cancel_event):
        try: self._execute_process_flow(params, cancel_event)
        finally: self.processing_finished()

    def start_batch_processing(self, base_params, cancel_event):
        try:
            if not self.available_regions: print("❌ No hay regiones."); return
            print(f"📋 {len(self.available_regions)} regiones para procesar.")
            for i, reg in enumerate(self.available_regions):
                if cancel_event.is_set(): break
                print(f"\n\n>>> 🌍 PROCESANDO REGIÓN {i+1}/{len(self.available_regions)}: {reg} <<<")
                cur_params = base_params.copy(); cur_params['aoi_path'] = BASE_GEOJSON_PATH / f"{reg}.geojson"
                self._execute_process_flow(cur_params, cancel_event)
        finally: self.processing_finished()

    def request_cancellation(self):
        print("\n--- 🛑 SOLICITUD DE CANCELACIÓN ---"); self.cancel_event.set(); self.cancel_button.config(state="disabled")

    def _get_month_list(self, y1, m1, y2, m2):
        months = []; f_ini, f_fin = datetime(y1, m1, 1), datetime(y2, m2, 1)
        if f_ini > f_fin: raise ValueError("Fecha inicio posterior a fecha fin.")
        c_y, c_m = y1, m1
        while True:
            months.append((c_y, c_m))
            if c_y == y2 and c_m == m2: break
            c_m += 1
            if c_m > 12: c_m, c_y = 1, c_y + 1
        return months

    def get_params(self, ignore_region=False):
        params = {}
        
        # Identificar Gas Seleccionado o Modo Ratio
        current_tab = self.gas_notebook.index("current")
        if current_tab == 0:
            params['gas_info'] = GAS_CONFIG[self.combo_sub_blh.get()]
            params['is_ratio'] = False
        elif current_tab == 1:
            params['gas_info'] = GAS_CONFIG[self.combo_post_blh.get()]
            params['is_ratio'] = False
        elif current_tab == 2:
            params['gas_num_info'] = GAS_CONFIG[self.combo_ratio_num.get()]
            params['gas_den_info'] = GAS_CONFIG[self.combo_ratio_den.get()]
            params['is_ratio'] = True
            params['gas_info'] = params['gas_num_info'] # Fallback
            
        gas_name = params['gas_info']['nombre_corto']
        
        choice = self.date_choice.get()
        params['choice_mode'] = choice
        mes_num_map = {name: num for num, name in meses_dict.items()}
        
        if choice == "mes":
            mes_nombre = self.mes_combo.get(); mes_num = mes_num_map[mes_nombre]; ano = int(self.ano_spin.get())
            _, last_day = calendar.monthrange(ano, mes_num)
            params['start_date'], params['end_date'] = f"{ano}-{mes_num:02d}-01", f"{ano}-{mes_num:02d}-{last_day}"
            params['year'], params['month'] = ano, mes_num
            params['title_date'] = meses_es_lower[mes_num]
            params['output_name'] = f"{mes_num:02d}_Datos_{gas_name}_{params['title_date']}_{ano}"
            params['output_name_blh'] = f"{mes_num:02d}_Datos_BLH_{params['title_date']}_{ano}"
            params['month_list'], params['title_suffix'] = [(ano, mes_num)], f"{mes_nombre} {ano}"
            params['folder_period'] = mes_nombre
            
        elif choice == "dia":
            fecha = self.dia_cal.get_date().strftime('%Y-%m-%d')
            params['start_date'] = params['end_date'] = fecha; params['year'], params['title_date'] = int(fecha[:4]), fecha
            params['output_name'], params['title_suffix'] = f"Datos_{gas_name}_Dia_{fecha}", f"Dia {fecha}"
            params['folder_period'] = fecha

        elif choice == "rango":
            start_date_obj, end_date_obj = self.rango_cal1.get_date(), self.rango_cal2.get_date()
            if start_date_obj > end_date_obj: raise ValueError("Inicio no puede ser posterior al fin.")
            start_date, end_date = start_date_obj.strftime('%Y-%m-%d'), end_date_obj.strftime('%Y-%m-%d')
            params['start_date'], params['end_date'], params['year'], params['month'] = start_date, end_date, int(start_date[:4]), start_date_obj.month 
            params['title_date'], params['title_suffix'] = f"Rango de {start_date} a {end_date}", f"Rango {start_date} a {end_date}"
            params['output_name'], params['output_name_blh'] = f"Datos_{gas_name}_Rango_{start_date}_a_{end_date}", f"Datos_BLH_Rango_{start_date}_a_{end_date}"
            params['folder_period'] = f"{start_date}_a_{end_date}"

        elif choice == "anio":
            ano = int(self.ano_spin_completo.get())
            params['year'], params['month_list'], params['title_suffix'] = ano, [(ano, m) for m in range(1, 13)], f"Año {ano}"
            params['folder_period'] = "Anio_Completo"

        elif choice == "rango_meses":
            m1, y1 = mes_num_map[self.rango_mes_combo_ini.get()], int(self.rango_ano_spin_ini.get())
            m2, y2 = mes_num_map[self.rango_mes_combo_fin.get()], int(self.rango_ano_spin_fin.get())
            params['month_list'], params['year'], params['title_suffix'] = self._get_month_list(y1, m1, y2, m2), y1, f"Rango {self.rango_mes_combo_ini.get()} {y1} - {self.rango_mes_combo_fin.get()} {y2}"
            params['folder_period'] = f"{self.rango_mes_combo_ini.get()}_{self.rango_mes_combo_fin.get()}"
            
        if not ignore_region:
            if self.region_mode.get() == "list":
                if not self.region_combo.get(): raise ValueError("Seleccione una región.")
                params['aoi_path'] = BASE_GEOJSON_PATH / f"{self.region_combo.get()}.geojson"
                if not params['aoi_path'].exists(): raise FileNotFoundError("GeoJSON no encontrado.")
            else:
                nm = "".join([c for c in self.manual_name_var.get().strip() if c.isalnum() or c in (' ', '_', '-')]).replace(" ", "_")
                if not nm: raise ValueError("Ingrese un nombre.")
                min_x, min_y, max_x, max_y = self.min_lon.get(), self.min_lat.get(), self.max_lon.get(), self.max_lat.get()
                if min_x >= max_x or min_y >= max_y: raise ValueError("Mínimos deben ser menores a máximos.")
                archivo_salida = BASE_GEOJSON_PATH / f"Manual_{nm}.geojson"
                gpd.GeoDataFrame({'geometry': [box(min_x, min_y, max_x, max_y)]}, crs="EPSG:4326").to_file(archivo_salida, driver="GeoJSON")
                params['aoi_path'] = archivo_salida
        else: params['aoi_path'] = None 

        params.update({
            "estadisticas": self.do_stats.get(), "generar_mapas": self.do_maps.get(), "do_zip": self.do_zip.get(),
            "show_plots": self.show_plots.get(), "palette_gas": self.palette_gas_combo.get(),
            "palette_blh": self.palette_blh_combo.get(), "palette_transform": self.palette_transform_combo.get(),
            "fig_queue": self.fig_queue, "formato_salida": self.formato_salida_var.get(),
            "do_regrid": self.do_regrid.get(), "do_comparative_map": self.do_comparative_map.get(),
            "generar_serie": self.do_timeseries.get()
        })

        params['transform_method'], params['transform_value'], params['descargar_blh'] = None, None, False 

        if self.do_transform.get() and not params['is_ratio']:
            selection = self.transform_method_combo.get()
            if "Petetin" in selection: params['transform_method'], params['descargar_blh'] = "petetin", True 
            elif "Savanets" in selection: params['transform_method'] = "savanets"
            elif "Valor escrito" in selection:
                val = simpledialog.askfloat("Altura BLH", "Ingrese altura en metros:\n(Ej: 1000)", parent=self.root, minvalue=1.0, maxvalue=50000.0)
                if val is None: return None 
                params['transform_method'], params['transform_value'] = "custom", val
        
        return params

    def _execute_process_flow(self, params, cancel_event):
        try:
            region_nombre = Path(params['aoi_path']).stem
            if params.get('is_ratio'):
                if params.get('choice_mode') in ['anio', 'rango_meses']:
                    ratio_paths, csvs_num, csvs_den = [], [], []
                    for year, month in params['month_list']:
                        if cancel_event.is_set(): break 
                        p_m = params.copy()
                        mes_nombre_es = meses_es_lower[month]
                        _, last_day = calendar.monthrange(year, month)
                        p_m.update({'start_date': f"{year}-{month:02d}-01", 'end_date': f"{year}-{month:02d}-{last_day}", 'year': year, 'month': month, 'title_date': mes_nombre_es, 'output_name': f"{month:02d}_Datos_{params['gas_num_info']['nombre_corto']}_{mes_nombre_es}_{year}"})
                        res = run_processing_ratio(p_m, cancel_event)
                        if res and res[0]:
                            ratio_paths.append(res[0]); csvs_num.append(res[1]); csvs_den.append(res[2])
                        
                    out_dir_ratio = BASE_OUTPUT_PATH / f"Ratios/{params['gas_num_info']['nombre_corto']}_{params['gas_den_info']['nombre_corto']}/{params['year']}/{region_nombre}/{params['folder_period']}"
                    
                    if params['do_comparative_map'] and not cancel_event.is_set():
                        cmap = "Spectral_r" if params['palette_gas'] in ['viridis', 'plasma', 'inferno'] else params['palette_gas']
                        if ratio_paths and (fig := generar_mapa_comparativo(ratio_paths, params['aoi_path'], f"Ratio {params['gas_num_info']['nombre_corto']}/{params['gas_den_info']['nombre_corto']}", "(adim.)", cmap, params['title_suffix'], output_dir=out_dir_ratio, return_fig=params['show_plots'])):
                            if params['show_plots']: params['fig_queue'].put(fig)
                            
                    if params['generar_serie'] and csvs_num and csvs_den and not cancel_event.is_set():
                        out_dir_ratio.mkdir(parents=True, exist_ok=True)
                        fig_dual = graficar_serie_temporal_dual(csvs_num, csvs_den, region_nombre, params['gas_num_info']['nombre_corto'], params['gas_den_info']['nombre_corto'], params['title_suffix'], out_dir_ratio, return_fig=params['show_plots'])
                        if fig_dual and params['show_plots']: params['fig_queue'].put(fig_dual)
                else:
                    res = run_processing_ratio(params, cancel_event)
                    if res and res[0] and params['generar_serie'] and not cancel_event.is_set():
                        out_dir_ratio = BASE_OUTPUT_PATH / f"Ratios/{params['gas_num_info']['nombre_corto']}_{params['gas_den_info']['nombre_corto']}/{params['year']}/{region_nombre}/{params['folder_period']}"
                        out_dir_ratio.mkdir(parents=True, exist_ok=True)
                        fig_dual = graficar_serie_temporal_dual([res[1]], [res[2]], region_nombre, params['gas_num_info']['nombre_corto'], params['gas_den_info']['nombre_corto'], params['title_suffix'], out_dir_ratio, return_fig=params['show_plots'])
                        if fig_dual and params['show_plots']: params['fig_queue'].put(fig_dual)
            else:
                if params.get('choice_mode') in ['anio', 'rango_meses']:
                    gas_name = params['gas_info']['nombre_corto']
                    print(f"\n==========================================================")
                    print(f"🚀 PROCESO MULTI-MES: {params['title_suffix']} | GAS: {gas_name}")
                    print(f"==========================================================")
                    
                    b_params, no2_paths, blh_paths, conc_paths, csv_paths, csv_paths_ppb = params.copy(), [], [], [], [], []
                    for year, month in b_params['month_list']:
                        if cancel_event.is_set(): break 
                        p_m = b_params.copy()
                        mes_nombre_es = meses_es_lower[month]
                        _, last_day = calendar.monthrange(year, month)
                        p_m.update({'start_date': f"{year}-{month:02d}-01", 'end_date': f"{year}-{month:02d}-{last_day}", 'year': year, 'month': month, 'title_date': mes_nombre_es, 'output_name': f"{month:02d}_Datos_{gas_name}_{mes_nombre_es}_{year}"})
                        p_m['output_name_blh'] = f"{month:02d}_Datos_BLH_{mes_nombre_es}_{year}" if p_m['descargar_blh'] else f"Datos_Transformados_{p_m.get('transform_method')}"
                        
                        n_p, b_p, c_p, csv_p, csv_ppb = run_processing(p_m, cancel_event)
                        if n_p: no2_paths.append(n_p)
                        if b_p: blh_paths.append(b_p)
                        if c_p: conc_paths.append(c_p)
                        if csv_p: csv_paths.append(csv_p)
                        if csv_ppb: csv_paths_ppb.append(csv_ppb)
                    
                    if params['do_comparative_map'] and not cancel_event.is_set():
                        fq = params['fig_queue'] if params['show_plots'] else None
                        
                        out_dir_col = BASE_OUTPUT_PATH / f"Modelo_{gas_name}/{params['year']}/{region_nombre}/{params['folder_period']}"
                        out_dir_col.mkdir(parents=True, exist_ok=True)
                        if no2_paths and (fig := generar_mapa_comparativo(no2_paths, params['aoi_path'], f"{gas_name} (Columna)", "(mol/m²)", params['palette_gas'], params['title_suffix'], output_dir=out_dir_col, return_fig=params['show_plots'])):
                            if fq: fq.put(fig)
                            
                        out_dir_blh = BASE_OUTPUT_PATH / f"BLH/{params['year']}/{region_nombre}/{params['folder_period']}"
                        out_dir_blh.mkdir(parents=True, exist_ok=True)
                        if blh_paths and params['descargar_blh'] and (fig := generar_mapa_comparativo(blh_paths, params['aoi_path'], "Capa Límite (BLH)", "(m)", params['palette_blh'], params['title_suffix'], output_dir=out_dir_blh, return_fig=params['show_plots'])):
                            if fq: fq.put(fig)
                            
                        out_dir_ppb = BASE_OUTPUT_PATH / f"Calculos_{gas_name}/{params['year']}/{region_nombre}/{params['folder_period']}"
                        out_dir_ppb.mkdir(parents=True, exist_ok=True)
                        if conc_paths and params.get('transform_method') and (fig := generar_mapa_comparativo(conc_paths, params['aoi_path'], f"Concentración {gas_name}", "(ppb)", params['palette_transform'], params['title_suffix'], output_dir=out_dir_ppb, return_fig=params['show_plots'])):
                            if fq: fq.put(fig)

                    if params['estadisticas'] and params['do_comparative_map'] and not cancel_event.is_set():
                        for p in no2_paths + blh_paths + conc_paths:
                            if p and (fig := analyze_tiff_statistics(str(p), params['show_plots'])) and params['show_plots']: params['fig_queue'].put(fig)
                            
                    if params['generar_serie'] and not cancel_event.is_set():
                        folder_period = params.get('folder_period', 'Periodo')
                        if csv_paths:
                            out_dir_col = BASE_OUTPUT_PATH / f"Modelo_{gas_name}/{params['year']}/{region_nombre}/{folder_period}"
                            out_dir_col.mkdir(parents=True, exist_ok=True)
                            fig_serie = graficar_serie_temporal(csv_paths, region_nombre, gas_name, params['title_suffix'], out_dir_col, is_ppb=False, return_fig=params['show_plots'])
                            if fig_serie and params['show_plots']: params['fig_queue'].put(fig_serie)
                        if csv_paths_ppb:
                            out_dir_ppb = BASE_OUTPUT_PATH / f"Calculos_{gas_name}/{params['year']}/{region_nombre}/{folder_period}"
                            out_dir_ppb.mkdir(parents=True, exist_ok=True)
                            fig_serie_ppb = graficar_serie_temporal(csv_paths_ppb, region_nombre, gas_name, params['title_suffix'], out_dir_ppb, is_ppb=True, return_fig=params['show_plots'])
                            if fig_serie_ppb and params['show_plots']: params['fig_queue'].put(fig_serie_ppb)

                else:
                    gas_name = params['gas_info']['nombre_corto']
                    n_p, b_p, c_p, csv_p, csv_ppb = run_processing(params, cancel_event)
                    if params['generar_serie'] and not cancel_event.is_set():
                        folder_period = params.get('folder_period', 'Periodo')
                        if csv_p:
                            out_dir_col = BASE_OUTPUT_PATH / f"Modelo_{gas_name}/{params['year']}/{region_nombre}/{folder_period}"
                            out_dir_col.mkdir(parents=True, exist_ok=True)
                            fig_serie = graficar_serie_temporal([csv_p], region_nombre, gas_name, params['title_suffix'], out_dir_col, is_ppb=False, return_fig=params['show_plots'])
                            if fig_serie and params['show_plots']: params['fig_queue'].put(fig_serie)
                        if csv_ppb:
                            out_dir_ppb = BASE_OUTPUT_PATH / f"Calculos_{gas_name}/{params['year']}/{region_nombre}/{folder_period}"
                            out_dir_ppb.mkdir(parents=True, exist_ok=True)
                            fig_serie_ppb = graficar_serie_temporal([csv_ppb], region_nombre, gas_name, params['title_suffix'], out_dir_ppb, is_ppb=True, return_fig=params['show_plots'])
                            if fig_serie_ppb and params['show_plots']: params['fig_queue'].put(fig_serie_ppb)

        except Exception as e: print(f"\n❌ ERROR INESPERADO: {e}")

    def processing_finished(self):
        print("\n\n🛑 DETENIDO!" if self.cancel_event.is_set() else "\n\n✅ FINALIZADO!")
        self.run_button.config(state="normal"); self.btn_download_all.config(state="normal"); self.cancel_button.config(state="disabled"); self.cancel_event.clear()

    def check_fig_queue(self):
        try: self.display_figure(self.fig_queue.get_nowait())
        except queue.Empty: pass
        self.root.after(100, self.check_fig_queue)

    def display_figure(self, fig):
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Visualizador")
        plot_window.geometry("1200x800")
        canvas = FigureCanvasTkAgg(fig, master=plot_window); canvas.draw()
        NavigationToolbar2Tk(canvas, plot_window).update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def clear_console(self):
        self.console.config(state='normal'); self.console.delete('1.0', tk.END); self.console.config(state='disabled')

    class TextRedirector:
        def __init__(self, app): self.app = app
        def write(self, str_):
            try: self.app.console.config(state='normal'); self.app.console.insert(tk.END, str_); self.app.console.see(tk.END); self.app.console.config(state='disabled')
            except tk.TclError: pass
        def flush(self): pass

# ==============================================================================
# SECCIÓN 3: PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    if not BASE_GEOJSON_PATH.is_dir():
        messagebox.showerror("Error de Configuración", f"No se encontró la carpeta 'Regiones'.\nAsegúrate que exista en:\n{SCRIPT_DIR}")
        sys.exit(1)
    available_regions = get_available_regions()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass 
    root = tk.Tk()
    app = GeoApp(root, available_regions)
    root.mainloop()

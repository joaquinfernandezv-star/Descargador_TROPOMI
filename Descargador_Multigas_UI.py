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
from datetime import datetime

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
    import contextily as cx
    from shapely.geometry import box
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

# --- RUTAS DINÁMICAS ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_OUTPUT_PATH = SCRIPT_DIR / "Resultados"
BASE_GEOJSON_PATH = SCRIPT_DIR / "Regiones"

BASE_OUTPUT_PATH.mkdir(exist_ok=True)
BASE_GEOJSON_PATH.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DE GASES (NUEVA ESTRUCTURA) ---
# Definimos las constantes físicas y de comportamiento para cada gas soportado.
GAS_CONFIG = {
    "NO2": {
        "nombre_corto": "NO2",
        "nombre_largo": "Dióxido de Nitrógeno",
        "band_name": "NO2", # Nombre de la banda en SentinelHub
        "peso_molecular": 46.01, # g/mol
        "factor_ppb": 24.45, # Factor base para conversión (volumen molar std)
        "comportamiento_vertical": "PBL_confined"
    },
    "O3": {
        "nombre_corto": "O3",
        "nombre_largo": "Ozono",
        "band_name": "O3",
        "peso_molecular": 48.00,
        "factor_ppb": 24.45,
        "comportamiento_vertical": "Thick_troposphere"
    },
    "SO2": {
        "nombre_corto": "SO2",
        "nombre_largo": "Dióxido de Azufre",
        "band_name": "SO2",
        "peso_molecular": 64.07,
        "factor_ppb": 24.45,
        "comportamiento_vertical": "PBL_confined"
    },
    "CO": {
        "nombre_corto": "CO",
        "nombre_largo": "Monóxido de Carbono",
        "band_name": "CO",
        "peso_molecular": 28.01,
        "factor_ppb": 24.45,
        "comportamiento_vertical": "Thick_troposphere"
    },
    "CH4": {
        "nombre_corto": "CH4",
        "nombre_largo": "Metano",
        "band_name": "CH4",
        "peso_molecular": 16.04,
        "factor_ppb": 24.45,
        "comportamiento_vertical": "Thick_troposphere"
    },
    "HCHO": {
        "nombre_corto": "HCHO",
        "nombre_largo": "Formaldehído",
        "band_name": "HCHO",
        "peso_molecular": 30.03,
        "factor_ppb": 24.45,
        "comportamiento_vertical": "PBL_confined"
    }
}

# --- CONFIGURACIÓN DE SENTINEL HUB ---
try:
    config = SHConfig()
    if not config.sh_client_id or not config.sh_client_secret:
        print("Configurando credenciales de Copernicus Data Space...")
        # NOTA: Asegúrate de que estas credenciales sean válidas o usa variables de entorno
        config.sh_client_id = "sh-2279fd56-dabb-4e4d-ae5b-b71ce5fc5c09"
        config.sh_client_secret = "9c94Zs5JMkwIkwGqyBJGCSXigh9jslVP"
        config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.save("cdse")
    print("Configuración de SentinelHub cargada.")
except Exception as e:
    print(f"Error configurando SentinelHub: {e}")

# Colección Genérica (S5P contiene todos los gases L2)
data_5p = DataCollection.SENTINEL5P.define_from("5p", service_url=config.sh_base_url)

# --- GENERADOR DE EVALSCRIPTS DINÁMICOS ---
def get_evalscript(gas_band, mode='simple'):
    """
    Genera el evalscript dinámicamente dependiendo del gas (banda) y el modo.
    gas_band: str ("NO2", "CO", "O3", etc.)
    mode: str ("simple" o "mean_mosaic")
    """
    if mode == 'mean_mosaic':
        # Evalscript para promedios (Mosaicking ORBIT)
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: ["{gas_band}", "dataMask"],
                output: {{
                    bands: 1,
                    sampleType: "FLOAT32",
                }},
                mosaicking: "ORBIT"
            }};
        }}

        function isClear(sample) {{
            return sample.dataMask == 1;
        }}

        function sum(array) {{
            let sum = 0;
            for (let i = 0; i < array.length; i++) {{
                sum += array[i].{gas_band};
            }}
            return sum;
        }}

        function evaluatePixel(samples) {{
            const clearTs = samples.filter(isClear)
            if (clearTs.length == 0) return [NaN];
            const mean = sum(clearTs) / clearTs.length
            return [mean]
        }}
        """
    else:
        # Evalscript simple (Raw/Día puntual)
        return f"""
        //VERSION=3
        function setup() {{ return {{ input: ["{gas_band}"], output: {{ bands: 1, sampleType: "FLOAT32" }}, mosaicking: "SIMPLE" }}; }}
        function evaluatePixel(samples) {{ return [samples.{gas_band}]; }}
        """

# --- Evalscript para NUBOSIDAD (Fijo) ---
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
    print(f"☁️  Analizando nubosidad para {time_start} - {time_end}...")
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None

    request_cloud = SentinelHubRequest(
        evalscript=evalscript_cloud,
        input_data=[SentinelHubRequest.input_data(
            data_collection=data_5p,
            time_interval=(time_start, time_end),
            other_args={'processing': {'upsampling': 'NEAREST', 'minQa': '50', 'timeliness': 'OFFL'}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=BBox(bbox=aoi_coords, crs=CRS.WGS84).transform(CRS(3857)),
        resolution=(2000, 2000), 
        config=config
    )
    
    try:
        data = request_cloud.get_data()[0]
        cloud_values = data.flatten()
        valid_clouds = cloud_values[~np.isnan(cloud_values)]
        
        if valid_clouds.size == 0: return 0.0
            
        mean_fraction = np.mean(valid_clouds)
        mean_percent = mean_fraction * 100
        print(f"✅ Nubosidad calculada: {mean_percent:.2f}%")
        return mean_percent
    except Exception as e:
        print(f"❌ Error calculando nubosidad: {e}")
        return None

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---

def datos_mes_gas(time_start, time_end, route, output_name, gas_config, evalscript_override=None):
    """
    Descarga datos del gas especificado.
    Adapta el evalscript a la banda del gas si no se provee override.
    """
    area = Path(route).stem
    aoi_coords = geojson_to_coords(route)
    if not aoi_coords: return None
    
    year_folder = time_start[:4]
    # Estructura: Modelo/Año/Area/Nombre_Salida
    data_folder = BASE_OUTPUT_PATH / f"Modelo/{year_folder}/{area}/{output_name}"
    data_folder.mkdir(parents=True, exist_ok=True)
    
    # Seleccionar evalscript: Si no hay override, usamos el raw (simple)
    if evalscript_override:
        script_to_use = evalscript_override
    else:
        script_to_use = get_evalscript(gas_config['band_name'], mode='simple')
    
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
    print("🛰️  Conectando al Copernicus CDS para descargar BLH (Promedio Mensual)...")
    month_str = f"{month:02d}"
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {'product_type': 'monthly_averaged_reanalysis', 'variable': 'boundary_layer_height', 
             'year': str(year), 'month': month_str,
             'time': '00:00', 'area': area_coords, 'format': 'netcdf'},
            output_path)
        print(f"✅ Descarga BLH Mensual completada: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error durante la descarga de BLH mensual desde CDS: {e}")
        return False

def descargar_blh_era5_diario(start_date, end_date, area_coords, output_path):
    print(f"📡 Solicitando ERA5 Horario (13:00) para {start_date}...")
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis', 'variable': 'boundary_layer_height',
                'year': str(start_date.year), 'month': f"{start_date.month:02d}", 'day': f"{start_date.day:02d}",
                'time': '13:00', 'area': area_coords, 'format': 'netcdf',
            },
            output_path
        )
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
        print(f"🔄 Archivo NetCDF (BLH) convertido a GeoTIFF: {tiff_path}")
        return True
    except Exception as e:
        print(f"❌ Error al convertir NetCDF (BLH) a TIFF: {e}")
        return False

def tag_nc_with_crs(netcdf_path, output_nc_path):
    try:
        data_array = rioxarray.open_rasterio(netcdf_path, variable='blh').squeeze()
        data_array.rio.write_crs("EPSG:4326", inplace=True)
        data_array.rio.to_raster(output_nc_path, driver='NETCDF')
        print(f"🏷️  Archivo BLH NetCDF etiquetado con CRS: {output_nc_path}")
        return True
    except Exception as e:
        print(f"❌ Error al etiquetar NetCDF (BLH) con CRS: {e}")
        return False

# --- FUNCIÓN GENERALIZADA DE CONVERSIÓN ---
def procesar_concentracion_gas(ruta_columna_cruda, ruta_blh_crudo, output_dir, region_nombre, anio, mes_nombre_es, formato_salida, metodotransform, valorcustom, suffix, gas_config, shape_factor_pbl=None):
    """
    Calcula la concentración superficial para el gas especificado.
    Adapta la lógica física según 'comportamiento_vertical' del gas.
    """
    nombre_gas_corto = gas_config['nombre_corto']
    nombre_gas_largo = gas_config['nombre_largo']
    
    print(f"\n---\n🔬 Iniciando cálculo de concentración de {nombre_gas_largo} ({nombre_gas_corto})")
    print(f"    Período: {mes_nombre_es} {anio} | Modo H: {metodotransform}")
    
    try:
        with rasterio.open(ruta_columna_cruda) as src_col:
            profile, ccol_array = src_col.profile, src_col.read(1).astype(np.float32)
            if src_col.nodata is not None: ccol_array[ccol_array == src_col.nodata] = np.nan
            
            h_resampled = None

            # 1. DETERMINAR H (Altura de Mezcla)
            if metodotransform == "petetin":
                if not ruta_blh_crudo:
                    print("❌ Error: Se seleccionó modo Petetin pero no se proporcionó archivo BLH.")
                    return None
                print(f"    Usando BLH dinámico desde: {Path(ruta_blh_crudo).name}")
                with rasterio.open(ruta_blh_crudo) as src_blh:
                    h_resampled = np.empty_like(ccol_array)
                    reproject(
                        source=rasterio.band(src_blh, 1), destination=h_resampled,
                        src_transform=src_blh.transform, src_crs=src_blh.crs,
                        dst_transform=src_col.transform, dst_crs=src_col.crs,
                        resampling=Resampling.bilinear
                    )
                    h_resampled[h_resampled <= 0] = np.nan

            elif metodotransform == "savanets":
                print("    Usando constante Savanets: 10,000 metros.")
                h_resampled = 10000.0
            
            elif metodotransform == "custom":
                if valorcustom is None:
                    print("❌ Error: Modo Custom sin valor.")
                    return None
                print(f"    Usando valor personalizado: {valorcustom} metros.")
                h_resampled = float(valorcustom)
            
            else:
                print(f"❌ Error: Método desconocido: {metodotransform}")
                return None
            
            # 2. LÓGICA FÍSICA SEGÚN TIPO DE GAS
            tipo_gas = gas_config.get("comportamiento_vertical", "PBL_confined")
            
            columna_para_calculo = ccol_array # Por defecto (PBL_confined) es la columna total
            
            if tipo_gas == "Thick_troposphere":
                # Para CO, CH4, O3, gran parte de la columna está por encima de la PBL.
                # Aplicamos un shape_factor para estimar solo la parte dentro de la PBL.
                if shape_factor_pbl is None:
                    print(f"⚠️  Advertencia: Gas '{nombre_gas_corto}' es de tipo 'Thick_troposphere' y no se definió shape_factor_pbl.")
                    print("    Se asumirá factor = 1.0 (potencial sobreestimación).")
                    shape_factor_pbl = 1.0
                else:
                    print(f"    Aplicando Shape Factor PBL: {shape_factor_pbl} para corregir columna troposférica.")
                
                columna_para_calculo = ccol_array * float(shape_factor_pbl)
            
            # 3. CÁLCULO ESTEQUIOMÉTRICO
            # Formula base: C_surf (ug/m3) = (C_col / H) * M * A
            # Donde A = 1e6 (conversión mol/m2 -> umol/m2 -> ug/m3 con volumen) ? No, es factor de unidad de área.
            # M = Masa Molar
            
            M = gas_config['peso_molecular']
            A = 1e6 # Factor constante (mol/m2 a umol/m2 si M está en g/mol... chequeo dimensional estándar)
            # Factor PPB: 24.45 / M para condiciones STP
            FACTOR_PPB_CALC = gas_config['factor_ppb'] / M 
            
            with np.errstate(divide='ignore', invalid='ignore'):
                concentracion_ug_m3 = (columna_para_calculo / h_resampled) * M * A
                concentracion_ppb = concentracion_ug_m3 * FACTOR_PPB_CALC
                
            if profile.get('nodata') is not None:
                concentracion_ppb[np.isnan(concentracion_ppb)] = profile['nodata']

            # 4. GUARDADO
            nombre_archivo_base = f"Concentracion_ppb_{nombre_gas_corto}_{region_nombre}_{anio}_{mes_nombre_es}{suffix}"

            if formato_salida == "NetCDF4":
                template_raster = rioxarray.open_rasterio(ruta_columna_cruda)
                data_con_banda = concentracion_ppb.astype(np.float32)[np.newaxis, :, :]
                data_array = template_raster.copy(data=data_con_banda)
                nombre_var_nc = f"concentracion_{nombre_gas_corto.lower()}_ppb"
                data_array = data_array.rename(nombre_var_nc)
                if profile.get('nodata') is not None:
                    data_array = data_array.rio.write_nodata(profile['nodata'])
                output_path = output_dir / f"{nombre_archivo_base}.nc"
                data_array.rio.to_raster(output_path, driver="NETCDF")

            elif formato_salida == "ASCII Grid (.asc)":
                output_path = output_dir / f"{nombre_archivo_base}.asc"
                nodata_value = -9999.0
                concentracion_para_asc = np.nan_to_num(concentracion_ppb, nan=nodata_value)
                profile.update(driver="AAIGrid", dtype=rasterio.float32, nodata=nodata_value)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(concentracion_para_asc.astype(rasterio.float32), 1)

            else: # GeoTIFF
                output_path = output_dir / f"{nombre_archivo_base}.tiff"
                profile.update(dtype=rasterio.float32)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(concentracion_ppb.astype(rasterio.float32), 1)

            print(f"✅ Cálculo de concentración completado: {output_path}")
            return output_path
            
    except Exception as e:
        print(f"❌ Error al procesar la concentración de {nombre_gas_corto}: {e}")
        return None

# --- COMPATIBILIDAD (WRAPPER NO2) ---
def procesar_concentracion_no2(ruta_no2_crudo, ruta_blh_crudo, output_dir, region_nombre, año, mes_nombre_es, formato_salida="GeoTIFF", metodo_transform="petetin", valor_custom=None, suffix=""):
    """Wrapper de compatibilidad para código antiguo que llame explícitamente a NO2."""
    return procesar_concentracion_gas(
        ruta_no2_crudo, ruta_blh_crudo, output_dir, region_nombre, año, mes_nombre_es,
        formato_salida, metodo_transform, valor_custom, suffix, 
        gas_config=GAS_CONFIG["NO2"]
    )

def regrid_geotiff(input_tiff_path, grid_resolution=100):
    """Rejilla (re-grid) un GeoTIFF usando interpolación Kriging."""
    carpeta = os.path.dirname(input_tiff_path)
    output_tiff_path = os.path.join(carpeta, "response_regrid.tiff")
    try:
        print(f"Cargando GeoTIFF: {input_tiff_path}")
        with rasterio.open(input_tiff_path) as ds:
            band1 = ds.read(1)
            transform = ds.transform; crs = ds.crs; nodata = ds.nodata
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
                        coords.append((lon, lat)); vals.append(val)
        if not vals: return None
        df = pd.DataFrame(coords, columns=["lon", "lat"]); df["value"] = vals
        lon_grid = np.linspace(df["lon"].min(), df["lon"].max(), grid_resolution)
        lat_grid = np.linspace(df["lat"].max(), df["lat"].min(), grid_resolution)
        ok = OrdinaryKriging(df["lon"].values, df["lat"].values, df["value"].values, variogram_model="spherical", verbose=False, enable_plotting=False)
        interpolado, _ = ok.execute("grid", lon_grid, lat_grid)
        res_x = (lon_grid.max() - lon_grid.min()) / (len(lon_grid) - 1) if grid_resolution > 1 else 0
        res_y = (lat_grid.min() - lat_grid.max()) / (len(lat_grid) - 1) if grid_resolution > 1 else 0
        left_edge = lon_grid.min() - res_x / 2; top_edge  = lat_grid.max() - res_y / 2 
        nuevo_transform = from_origin(left_edge, top_edge, res_x, abs(res_y))
        with rasterio.open(output_tiff_path, "w", driver="GTiff", height=interpolado.shape[0], width=interpolado.shape[1], count=1, dtype=interpolado.dtype, crs=crs, transform=nuevo_transform) as dst:
            dst.write(interpolado, 1)
        return output_tiff_path
    except Exception as e:
        print(f"Error Kriging: {e}"); return None

# --- FUNCIONES DE VISUALIZACIÓN ---
def generar_mapa_con_leyenda(ruta_tiff, ruta_geojson, title_date, year, cmap='viridis', alpha=0.75, producto="Producto", unidad="(unidad)", return_fig=False):
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
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        
        img = ax.imshow(data, extent=extent, cmap=cmap_obj, norm=norm, alpha=alpha, zorder=10)
        gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, linestyle='--', zorder=11)
        try: cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
        except Exception: pass

        ax.set_title(f"{producto} en {region_nombre} • {title_date.capitalize()} {year} • Datos {procesamiento}", fontsize=16, pad=15)
        ax.set_xlabel("Longitud", fontsize=14); ax.set_ylabel("Latitud", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.2)
        cbar = plt.colorbar(img, cax=cax); cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=20, fontsize=14)
        plt.tight_layout(); plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"🖼️ Mapa guardado: {output_path}")
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception as e: print(f"Error mapa: {e}"); return None

def generar_mapa_comparativo(file_paths, aoi_path, producto, unidad, cmap, title_suffix, return_fig=False):
    if not file_paths: return None
    try:
        gdf = gpd.read_file(aoi_path).dissolve()
        region_nombre = Path(aoi_path).stem.replace("_", " ").title()
        vmin, vmax = np.inf, -np.inf
        all_data, valid_extents = {}, []
        target_crs = None 
        
        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                if target_crs is None: target_crs = src.crs
                gdf_proj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
                try:
                    data, out_transform = mask(dataset=src, shapes=gdf_proj.geometry, crop=True, nodata=np.nan)
                    data = data[0]
                    if np.all(np.isnan(data)): all_data[file_path] = {'data': None}; continue
                    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], out_transform)
                    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
                    all_data[file_path] = {'data': data, 'extent': extent}
                    valid_extents.append(extent)
                    vmin = min(vmin, np.nanmin(data)); vmax = max(vmax, np.nanmax(data))
                except ValueError: all_data[file_path] = {'data': None}

        if not valid_extents: return None
        global_left = min(e[0] for e in valid_extents); global_right = max(e[1] for e in valid_extents)
        global_bottom = min(e[2] for e in valid_extents); global_top = max(e[3] for e in valid_extents)
        gdf_plot = gdf.to_crs(target_crs) if target_crs and gdf.crs != target_crs else gdf
        
        n = len(file_paths)
        if n <= 4: nrows, ncols = 1, n
        elif n <= 8: nrows, ncols = 2, (n + 1) // 2
        else: nrows, ncols = (n + 3) // 4, 4 
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 7)) 
        axs = np.atleast_1d(axs).flatten() 
        cmap_obj = cmc.batlow if tiene_cmcrameri and cmap == 'batlow' else plt.get_cmap(cmap)
        norm = Normalize(vmin=vmin, vmax=vmax)
        img = None 
        
        for i, file_path in enumerate(file_paths):
            ax = axs[i]; plot_data = all_data[file_path]
            title = file_path.stem
            try:
                parts = file_path.parent.name.split('_')
                if len(parts) >= 2 and parts[-1].isdigit(): title = f"{parts[-2].capitalize()} {parts[-1]}"
            except: pass
            
            if plot_data['data'] is not None:
                img = ax.imshow(plot_data['data'], extent=plot_data['extent'], cmap=cmap_obj, norm=norm, alpha=0.8, zorder=10)
                gdf_plot.boundary.plot(ax=ax, edgecolor='black', linewidth=1.0, linestyle='--', zorder=11)
                ax.set_title(title, fontsize=11)
                try: cx.add_basemap(ax, crs=gdf_plot.crs, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
                except: pass
            else:
                ax.set_title(f"{title}\n(Sin datos)", fontsize=12); ax.set_facecolor('0.95') 
            ax.set_xlim(global_left, global_right); ax.set_ylim(global_bottom, global_top)
            ax.tick_params(labelbottom=False, labelleft=False)
            if i // ncols == nrows - 1: ax.tick_params(labelbottom=True); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            if i % ncols == 0: ax.tick_params(labelleft=True)

        for i in range(n, len(axs)): axs[i].set_visible(False)
        fig.suptitle(f"{producto} en {region_nombre} • {title_suffix}", fontsize=18)
        
        if img:
            fig.tight_layout(rect=[0.04, 0.05, 0.85, 0.95])
            cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7]); cbar = fig.colorbar(img, cax=cbar_ax)
            cbar.set_label(f"{producto} {unidad}", rotation=270, labelpad=25, fontsize=14)
            fig.supxlabel('Longitud', fontsize=16, y=0.02); fig.supylabel('Latitud', fontsize=16, x=0.01)
        else: fig.tight_layout()

        output_path = file_paths[0].parent.parent / f"Mapa_Comparativo_{producto}_{title_suffix.replace(' ', '_')}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"🖼️  Mapa comparativo: {output_path}")
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception as e: print(f"Error mapa comparativo: {e}"); return None

def analyze_tiff_statistics(tiff_path: str, return_fig=False):
    try:
        with rasterio.open(tiff_path) as src:
            image_data = src.read(1).astype(np.float32)
            if src.nodata is not None: image_data[image_data == src.nodata] = np.nan
            pixel_values = image_data[~np.isnan(image_data)]
        if pixel_values.size == 0: return
        mean_val, std_val = np.nanmean(pixel_values), np.nanstd(pixel_values)
        median_val = np.nanmedian(pixel_values)
        
        stats_df = pd.DataFrame({
            'Statistic': ['Minimum', 'Maximum', 'Median', 'Mean', 'Std Dev', 'Data Points'],
            'Value': [np.nanmin(pixel_values), np.nanmax(pixel_values), f"{median_val:.3e}", f"{mean_val:.3e}", f"{std_val:.3e}", pixel_values.size]
        })
        csv_output_path = f"{Path(tiff_path).with_suffix('')}_statistics.csv"
        stats_df.to_csv(csv_output_path, index=False)

        fig = plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=50, density=True, alpha=0.7, color='skyblue', label='Distribución')
        x_norm = np.linspace(pixel_values.min(), pixel_values.max(), 100)
        pdf_fitted = norm.pdf(x_norm, mean_val, std_val)
        plt.plot(x_norm, pdf_fitted, 'r-', linewidth=2, label=f'Normal (μ={mean_val:.2e})')
        plt.title(f"Distribución - {Path(tiff_path).stem}"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
        plot_output_path = f"{Path(tiff_path).with_suffix('')}_distribution.png"
        plt.savefig(plot_output_path)
        if return_fig: return fig
        else: plt.close(fig); return None
    except Exception: return None

# --- FUNCIÓN PRINCIPAL DE PROCESAMIENTO (MODIFICADA PARA MULTIGAS) ---
def run_processing(params, cancel_event):
    matplotlib.use('Agg')
    
    aoi_path = params['aoi_path']
    region_nombre = Path(aoi_path).stem
    time_start, time_end = params['start_date'], params['end_date']
    
    # Obtener configuración del gas
    gas_config = params['gas_config']
    gas_short = gas_config['nombre_corto']
    gas_long = gas_config['nombre_largo']
    
    ruta_crudo_gas = None
    ruta_blh_para_calculo = None 
    ruta_gas_para_procesar = None
    ruta_blh_para_analisis = None
    ruta_concentracion_final = None

    print(f"\n\n--- 🗓️  PROCESANDO {gas_long} ({gas_short}) en {region_nombre.replace('_', ' ').title()} ---")
    print(f"    Período: {time_start} a {time_end}")

    # --- 1. PROCESO PARA GAS (SENTINEL-5P) ---
    print(f"\n📥 Descargando datos de {gas_short}...")
    
    use_mean_script = params['choice_mode'] in ['mes', 'anio', 'rango_meses', 'rango']
    # Generar script dinámico usando el nombre de banda del gas
    evalscript_mode = 'mean_mosaic' if use_mean_script else 'simple'
    script_to_use = get_evalscript(gas_config['band_name'], mode=evalscript_mode)

    print(f"ℹ️  Usando Evalscript modo: {evalscript_mode}")

    request_gas = datos_mes_gas(time_start, time_end, aoi_path, params['output_name'], gas_config, evalscript_override=script_to_use)
    
    if request_gas and request_gas.get_filename_list():
        carpeta_base = Path(request_gas.data_folder).resolve()
        ruta_crudo_gas = (carpeta_base / request_gas.get_filename_list()[0]).resolve()
        print(f"🗂️  Archivo base {gas_short} (crudo): {ruta_crudo_gas}")
        
        ruta_gas_para_procesar = ruta_crudo_gas 

        if params['do_regrid']:
            print(f"\n🔄 Iniciando re-escalado (Kriging)...")
            ruta_regrid_str = regrid_geotiff(str(ruta_crudo_gas)) 
            if ruta_regrid_str:
                ruta_gas_para_procesar = Path(ruta_regrid_str).resolve()
                print(f"✅ Re-escalado OK: {ruta_regrid_str}")
        
        if not params.get('do_comparative_map', False):
            if params['estadisticas']:
                if fig := analyze_tiff_statistics(str(ruta_gas_para_procesar), return_fig=params['show_plots']): params['fig_queue'].put(fig)
            if params['generar_mapas']:
                if fig := generar_mapa_con_leyenda(ruta_gas_para_procesar, aoi_path, params['title_date'], params['year'], cmap=params['palette'], producto=f"{gas_short} (Columna)", unidad="(mol/m²)", return_fig=params['show_plots']): params['fig_queue'].put(fig)
    else:
        print(f"⚠️  Advertencia: No se obtuvieron datos de {gas_short}.")

    if cancel_event.is_set(): return None, None, None

    # --- 2. PROCESO PARA BLH (ERA5) --- 
    if params.get('transform_method') == 'petetin':
        print(f"\n🌍 Descargando datos de BLH (H) para: {params['title_date']} {params['year']}")
        bounding_box = geojson_to_coords(aoi_path)
        if bounding_box:
            min_x, min_y, max_x, max_y = bounding_box
            era5_area_coords = [max_y, min_x, min_y, max_x]
            
            blh_output_dir = BASE_OUTPUT_PATH / f"BLH/{params['year']}/{region_nombre}/{params['output_name_blh']}"
            blh_output_dir.mkdir(parents=True, exist_ok=True)
            nombre_base_blh = f"BLH_ERA5_{region_nombre}_{params['year']}_{params['title_date']}"
            netcdf_file = blh_output_dir / f"{nombre_base_blh}.nc"
            tiff_file_blh = blh_output_dir / f"{nombre_base_blh}.tiff"

            exito_descarga = False
            if params.get('choice_mode') == 'dia':
                 try:
                     fecha_dt = datetime.strptime(params['start_date'], '%Y-%m-%d')
                     exito_descarga = descargar_blh_era5_diario(fecha_dt, fecha_dt, era5_area_coords, netcdf_file)
                 except Exception as e:
                     print(f"❌ Error fecha descarga diaria: {e}")
            else:
                 if 'month' in params:
                     exito_descarga = descargar_blh_era5(params['year'], params['month'], era5_area_coords, netcdf_file)

            if exito_descarga:
                netcdf_file_tagged = blh_output_dir / f"{nombre_base_blh}_tagged.nc"
                ruta_blh_para_analisis = None 
                
                if params['formato_salida'] == "GeoTIFF":
                    if convertir_nc_a_tiff(netcdf_file, tiff_file_blh):
                        ruta_blh_para_calculo = tiff_file_blh 
                        ruta_blh_para_analisis = tiff_file_blh
                else: 
                    if tag_nc_with_crs(netcdf_file, netcdf_file_tagged):
                        ruta_blh_para_calculo = netcdf_file_tagged
                        ruta_blh_para_analisis = netcdf_file_tagged
                
                if ruta_blh_para_analisis and not params.get('do_comparative_map', False):
                    if params['estadisticas']:
                         if fig := analyze_tiff_statistics(str(ruta_blh_para_analisis), return_fig=params['show_plots']): params['fig_queue'].put(fig)
    
    if cancel_event.is_set(): return ruta_gas_para_procesar, None, None

    # --- 3. CÁLCULO DE CONCENTRACIÓN ---
    can_calculate = False
    if ruta_gas_para_procesar and params.get('transform_method'):
        if params['transform_method'] == 'petetin': can_calculate = (ruta_blh_para_calculo is not None)
        else: can_calculate = True 
    
    if can_calculate:
        if params['transform_method'] == 'savanets': folder_leaf = "Savenets Equation"
        elif params['transform_method'] == 'custom': folder_leaf = "Custom transformed"
        else: folder_leaf = params['output_name_blh'].replace('BLH', 'Concentracion') if params.get('output_name_blh') else params['output_name']
        
        calc_output_dir = BASE_OUTPUT_PATH / f"Calculos/{params['year']}/{region_nombre}/{folder_leaf}"
        calc_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_suffix = "_regrid" if params['do_regrid'] and ruta_gas_para_procesar != ruta_crudo_gas else ""

        # USAR NUEVA FUNCIÓN GENÉRICA
        ruta_concentracion_final = procesar_concentracion_gas(
            ruta_gas_para_procesar, 
            ruta_blh_para_calculo,
            calc_output_dir, 
            region_nombre, 
            params['year'], 
            params['title_date'],
            params['formato_salida'],
            metodotransform=params['transform_method'],
            valorcustom=params.get('transform_value'),
            suffix=file_suffix,
            gas_config=gas_config,  # Pasamos configuración
            shape_factor_pbl=params.get('shape_factor_pbl') # Pasamos factor (puede ser None)
        )
        
        if ruta_concentracion_final and not params.get('do_comparative_map', False):
            if params['estadisticas']:
                if fig := analyze_tiff_statistics(str(ruta_concentracion_final), return_fig=params['show_plots']): params['fig_queue'].put(fig)
            if params['generar_mapas']:
                if fig := generar_mapa_con_leyenda(ruta_concentracion_final, aoi_path, params['title_date'], params['year'], cmap='inferno', producto=f"Concentración {gas_short}", unidad="(ppb)", return_fig=params['show_plots']): params['fig_queue'].put(fig)

    if cancel_event.is_set(): return ruta_gas_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final

    # --- 4. COMPRESIÓN ---
    if params['do_zip'] and ruta_crudo_gas:
        source_folder = Path(request_gas.data_folder).resolve()
        zip_destino = source_folder.parent / f"{source_folder.name}_{gas_short}-crudo.zip"
        comprimir_directorio(source_folder, zip_destino)

    return ruta_gas_para_procesar, ruta_blh_para_analisis, ruta_concentracion_final

# ==============================================================================
# SECCIÓN 2: CLASE DE LA APLICACIÓN TKINTER
# ==============================================================================
class GeoApp:
    def __init__(self, root, available_regions):
        self.root = root
        self.root.title("Descargador y Procesador Sentinel-5P Multigas")
        self.root.geometry("850x950")
        self.fig_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.available_regions = available_regions
        self.cancel_event = threading.Event()
        
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
        if self.cancel_event: self.cancel_event.set()
        sys.stdout = self.original_stdout
        self.root.destroy()
        
    def create_date_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="1. Selección de Fecha", padding="10")
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.date_choice = tk.StringVar(value="mes")
        
        radio_frame = ttk.Frame(frame)
        radio_frame.grid(row=0, column=0, columnspan=4, sticky="w")
        for val, text in [("mes", "Mes"), ("anio", "Año"), ("dia", "Día"), ("rango", "Rango Días"), ("rango_meses", "Rango Meses")]:
            ttk.Radiobutton(radio_frame, text=text, variable=self.date_choice, value=val, command=self.toggle_date_widgets).pack(side=tk.LEFT, padx=5)

        self.date_widgets_frame = ttk.Frame(frame)
        self.date_widgets_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)
        
        # Widgets fechas (reutilizados)
        self.mes_label = ttk.Label(self.date_widgets_frame, text="Mes:")
        self.mes_combo = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes)
        self.mes_combo.set("Enero")
        self.ano_label = ttk.Label(self.date_widgets_frame, text="Año:")
        self.ano_spin = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.ano_label_completo = ttk.Label(self.date_widgets_frame, text="Año:")
        self.ano_spin_completo = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.dia_label = ttk.Label(self.date_widgets_frame, text="Fecha:")
        self.dia_cal = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_label1 = ttk.Label(self.date_widgets_frame, text="Desde:"); self.rango_cal1 = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_label2 = ttk.Label(self.date_widgets_frame, text="Hasta:"); self.rango_cal2 = DateEntry(self.date_widgets_frame, date_pattern='yyyy-mm-dd', width=12)
        self.rango_mes_label_ini = ttk.Label(self.date_widgets_frame, text="Desde:"); self.rango_mes_combo_ini = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes); self.rango_ano_spin_ini = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.rango_mes_label_fin = ttk.Label(self.date_widgets_frame, text="Hasta:"); self.rango_mes_combo_fin = ttk.Combobox(self.date_widgets_frame, values=list(meses_dict.values()), state="readonly", width=self.ancho_combobox_mes); self.rango_ano_spin_fin = tk.Spinbox(self.date_widgets_frame, from_=2018, to=2030, width=self.ancho_spinbox_ano)
        self.rango_mes_combo_ini.set("Enero"); self.rango_mes_combo_fin.set("Marzo")

    def toggle_date_widgets(self):
        for widget in self.date_widgets_frame.winfo_children(): widget.grid_forget()
        choice = self.date_choice.get()
        if choice == "mes":
            self.mes_label.grid(row=0, column=0); self.mes_combo.grid(row=0, column=1); self.ano_label.grid(row=0, column=2); self.ano_spin.grid(row=0, column=3)
        elif choice == "anio":
            self.ano_label_completo.grid(row=0, column=0); self.ano_spin_completo.grid(row=0, column=1)
        elif choice == "dia":
            self.dia_label.grid(row=0, column=0); self.dia_cal.grid(row=0, column=1)
        elif choice == "rango":
            self.rango_label1.grid(row=0, column=0); self.rango_cal1.grid(row=0, column=1); self.rango_label2.grid(row=0, column=2); self.rango_cal2.grid(row=0, column=3)
        elif choice == "rango_meses":
            self.rango_mes_label_ini.grid(row=0, column=0); self.rango_mes_combo_ini.grid(row=0, column=1); self.rango_ano_spin_ini.grid(row=0, column=2)
            self.rango_mes_label_fin.grid(row=1, column=0); self.rango_mes_combo_fin.grid(row=1, column=1); self.rango_ano_spin_fin.grid(row=1, column=2)
        self.update_options_state()

    def create_region_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Selección de Región", padding="10")
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.region_mode = tk.StringVar(value="list")
        rb_frame = ttk.Frame(frame); rb_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(rb_frame, text="Lista Precargada", variable=self.region_mode, value="list", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(rb_frame, text="Coordenadas Manuales", variable=self.region_mode, value="manual", command=self.toggle_region_mode).pack(side=tk.LEFT, padx=10)
        
        self.region_container = ttk.Frame(frame); self.region_container.pack(fill=tk.X, expand=True, pady=5)
        self.frame_list = ttk.Frame(self.region_container)
        ttk.Label(self.frame_list, text="Región:").pack(side=tk.LEFT, padx=5)
        self.region_combo = ttk.Combobox(self.frame_list, values=self.available_regions, state="readonly", width=40); self.region_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.available_regions: self.region_combo.set(self.available_regions[0])
        
        self.frame_manual = ttk.Frame(self.region_container)
        nm_frame = ttk.Frame(self.frame_manual); nm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(nm_frame, text="Nombre Zona:").pack(side=tk.LEFT); self.manual_name_var = tk.StringVar(); ttk.Entry(nm_frame, textvariable=self.manual_name_var, width=30).pack(side=tk.LEFT, padx=5)
        coord_frame = ttk.Frame(self.frame_manual); coord_frame.pack(fill=tk.X, pady=5)
        self.min_lon = tk.DoubleVar(value=-71.0); self.min_lat = tk.DoubleVar(value=-33.6)
        self.max_lon = tk.DoubleVar(value=-70.0); self.max_lat = tk.DoubleVar(value=-33.0)
        # (Widgets de coordenadas omitidos por brevedad, son iguales a original)
        ttk.Label(coord_frame, text="W:").grid(row=0,column=0); ttk.Entry(coord_frame, textvariable=self.min_lon, width=8).grid(row=0,column=1)
        ttk.Label(coord_frame, text="E:").grid(row=0,column=2); ttk.Entry(coord_frame, textvariable=self.max_lon, width=8).grid(row=0,column=3)
        ttk.Label(coord_frame, text="S:").grid(row=1,column=0); ttk.Entry(coord_frame, textvariable=self.min_lat, width=8).grid(row=1,column=1)
        ttk.Label(coord_frame, text="N:").grid(row=1,column=2); ttk.Entry(coord_frame, textvariable=self.max_lat, width=8).grid(row=1,column=3)
        self.toggle_region_mode()

    def toggle_region_mode(self):
        self.frame_list.pack_forget(); self.frame_manual.pack_forget()
        if self.region_mode.get() == "list": self.frame_list.pack(fill=tk.X)
        else: self.frame_manual.pack(fill=tk.X)

    def create_options_widgets(self, parent):
        proc_frame = ttk.LabelFrame(parent, text="3. Opciones de Procesamiento", padding="10")
        proc_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # --- SELECCION DE GAS (NUEVO) ---
        gas_frame = ttk.Frame(proc_frame)
        gas_frame.pack(anchor="w", fill=tk.X, pady=(0, 5))
        ttk.Label(gas_frame, text="Gas a Procesar:", font=("Default", 9, "bold")).pack(side=tk.LEFT, padx=(0,5))
        self.gas_var = tk.StringVar(value="NO2")
        self.gas_combo = ttk.Combobox(gas_frame, textvariable=self.gas_var, values=list(GAS_CONFIG.keys()), state="readonly", width=10)
        self.gas_combo.pack(side=tk.LEFT)
        # Binding para actualizar opciones al cambiar gas
        self.gas_combo.bind("<<ComboboxSelected>>", self.update_options_state)
        # --------------------------------

        self.do_stats = tk.BooleanVar(value=True)
        self.do_maps = tk.BooleanVar(value=True)
        self.do_transform = tk.BooleanVar(value=True)
        self.do_zip = tk.BooleanVar()
        self.do_regrid = tk.BooleanVar(value=False)
        self.do_comparative_map = tk.BooleanVar(value=False)
        
        self.transform_checkbutton = ttk.Checkbutton(proc_frame, text="Transformar a superficie", variable=self.do_transform, command=self.toggle_transform_options)
        self.transform_checkbutton.pack(anchor="w")
        
        self.transform_options_frame = ttk.Frame(proc_frame); self.transform_options_frame.pack(anchor="w", padx=20)
        self.transform_method_var = tk.StringVar(value="Descargar BLH y Calcular Concentración (H. Petetin Mode)")
        self.transform_method_combo = ttk.Combobox(self.transform_options_frame, textvariable=self.transform_method_var, 
            values=["Descargar BLH y Calcular Concentración (H. Petetin Mode)", "Ecuación de Savanets (10km)", "Valor escrito por el usuario"], 
            state="readonly", width=45)
        self.transform_method_combo.pack(anchor="w")
        
        ttk.Checkbutton(proc_frame, text="Re-escalar datos (Kriging)", variable=self.do_regrid).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Analizar Estadísticas", variable=self.do_stats).pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Generar Mapas Individuales", variable=self.do_maps).pack(anchor="w")
        self.comp_map_checkbutton = ttk.Checkbutton(proc_frame, text="Generar Mapa Comparativo (Año/Rango)", variable=self.do_comparative_map)
        self.comp_map_checkbutton.pack(anchor="w")
        ttk.Checkbutton(proc_frame, text="Comprimir resultados", variable=self.do_zip).pack(anchor="w")
        
        ttk.Label(proc_frame, text="Formato Salida:").pack(anchor='w', pady=(5,0))
        self.formato_salida_var = tk.StringVar(value="GeoTIFF")
        ttk.Combobox(proc_frame, textvariable=self.formato_salida_var, values=["GeoTIFF", "NetCDF4", "ASCII Grid (.asc)"], state="readonly").pack(fill=tk.X, anchor='w')
        
        vis_frame = ttk.LabelFrame(parent, text="4. Visualización", padding="10")
        vis_frame.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.show_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Mostrar gráficos", variable=self.show_plots).pack(anchor="w")
        ttk.Label(vis_frame, text="Paleta de Colores:").pack(anchor='w', pady=(5,0))
        self.palette_combo = ttk.Combobox(vis_frame, values=list(paletas_colores.keys()), state="readonly"); self.palette_combo.pack(fill=tk.X); self.palette_combo.set("viridis")
        
        # Nubosidad
        ttk.Separator(vis_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(vis_frame, text="Verificación Nubosidad:", font=("Default", 9, "bold")).pack(anchor='w')
        cloud_frame = ttk.Frame(vis_frame); cloud_frame.pack(fill=tk.X)
        self.btn_cloud = ttk.Button(cloud_frame, text="☁️ Calc %", command=self.on_cloud_click); self.btn_cloud.pack(side=tk.LEFT)
        self.lbl_cloud_result = ttk.Label(cloud_frame, text="---", background="black", foreground="#00ff00", width=8, anchor="center"); self.lbl_cloud_result.pack(side=tk.LEFT, padx=5)

    def on_cloud_click(self):
        try:
            params = self.get_params()
            if not params: return
            self.btn_cloud.config(state="disabled"); self.lbl_cloud_result.config(text="...", foreground="yellow")
            threading.Thread(target=self.run_cloud_check, args=(params,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.btn_cloud.config(state="normal")

    def run_cloud_check(self, params):
        try:
            percent = calcular_estadisticas_nubosidad(params['start_date'], params['end_date'], params['aoi_path'])
            color = "#00ff00" if percent < 20 else ("orange" if percent < 50 else "red")
            self.root.after(0, lambda: [self.lbl_cloud_result.config(text=f"{percent:.1f}%", foreground=color), self.btn_cloud.config(state="normal")])
        except:
             self.root.after(0, lambda: self.btn_cloud.config(state="normal"))

    def toggle_transform_options(self):
        state = "readonly" if self.do_transform.get() else "disabled"
        self.transform_method_combo.config(state=state)

    def update_options_state(self, event=None):
        date_choice = self.date_choice.get()
        selected_gas = self.gas_var.get()
        
        # Lista negra para transformación
        gases_sin_transformacion = ["CO", "CH4", "O3"]
        
        # Lógica para habilitar/deshabilitar transformación
        # 1. Modo fecha compatible (mes, anio, etc.)
        date_compatible = date_choice in ['mes', 'anio', 'rango_meses', 'dia', 'rango']
        # 2. El gas NO debe estar en la lista negra
        gas_compatible = selected_gas not in gases_sin_transformacion
        
        if date_compatible and gas_compatible:
            self.transform_checkbutton.config(state="normal")
            if self.do_transform.get():
                self.transform_method_combo.config(state="readonly")
            else:
                self.transform_method_combo.config(state="disabled")
        else:
            self.transform_checkbutton.config(state="disabled")
            self.do_transform.set(False) # Forzar desactivado
            self.transform_method_combo.config(state="disabled")
        
        # Mapa comparativo lógica sigue igual
        state_map = "normal" if date_choice in ['anio', 'rango_meses'] else "disabled"
        self.comp_map_checkbutton.config(state=state_map)
        if state_map == "disabled": self.do_comparative_map.set(False)

    def create_console_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Consola", padding="10")
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        parent.grid_rowconfigure(3, weight=1); parent.grid_columnconfigure(0, weight=1)
        self.console = scrolledtext.ScrolledText(frame, state='disabled', height=18, bg="black", fg="white", font=("Courier New", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        sys.stdout = self.TextRedirector(self)

    def create_action_buttons(self, parent):
        frame = ttk.Frame(parent, padding="10"); frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(frame, text="Iniciar Proceso", command=self.start_processing_thread); self.run_button.pack(side=tk.RIGHT, padx=5)
        self.btn_download_all = ttk.Button(frame, text="Descargar Todo (Batch)", command=self.start_batch_processing_thread); self.btn_download_all.pack(side=tk.RIGHT, padx=5)
        self.cancel_button = ttk.Button(frame, text="Cancelar", command=self.request_cancellation, state="disabled"); self.cancel_button.pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame, text="Salir", command=self.on_closing).pack(side=tk.RIGHT)

    def start_processing_thread(self):
        self.cancel_event.clear()
        self.toggle_buttons(False)
        self.clear_console()
        try:
            params = self.get_params()
            if params: threading.Thread(target=self.start_single_processing, args=(params, self.cancel_event), daemon=True).start()
            else: self.processing_finished()
        except Exception as e: messagebox.showerror("Error", str(e)); self.processing_finished()

    def start_batch_processing_thread(self):
        self.cancel_event.clear()
        self.toggle_buttons(False)
        self.clear_console()
        try:
            base_params = self.get_params(ignore_region=True)
            if base_params: threading.Thread(target=self.start_batch_processing, args=(base_params, self.cancel_event), daemon=True).start()
            else: self.processing_finished()
        except Exception as e: messagebox.showerror("Error", str(e)); self.processing_finished()

    def toggle_buttons(self, enable):
        state = "normal" if enable else "disabled"
        self.run_button.config(state=state); self.btn_download_all.config(state=state)
        self.cancel_button.config(state="disabled" if enable else "normal")

    def request_cancellation(self):
        self.cancel_event.set(); self.cancel_button.config(state="disabled")
        print("\n🛑 Cancelación solicitada...")

    def start_single_processing(self, params, cancel_event):
        try: self._execute_process_flow(params, cancel_event)
        finally: self.processing_finished()

    def start_batch_processing(self, base_params, cancel_event):
        try:
            total = len(self.available_regions)
            if total == 0: print("❌ No hay regiones."); return
            print(f"📋 Procesando {total} regiones por lotes.")
            for i, reg in enumerate(self.available_regions):
                if cancel_event.is_set(): break
                print(f"\n>>> 🌍 Región {i+1}/{total}: {reg}")
                curr = base_params.copy()
                curr['aoi_path'] = BASE_GEOJSON_PATH / f"{reg}.geojson"
                self._execute_process_flow(curr, cancel_event)
            if not cancel_event.is_set(): print("\n✅ Batch completado.")
        except Exception as e: print(f"Error Batch: {e}")
        finally: self.processing_finished()

    def _get_month_list(self, y1, m1, y2, m2):
        months = []
        d = datetime(y1, m1, 1); end = datetime(y2, m2, 1)
        if d > end: raise ValueError("Fecha inicio posterior a fin.")
        curr = d
        while curr <= end:
            months.append((curr.year, curr.month))
            if curr.month == 12: curr = datetime(curr.year + 1, 1, 1)
            else: curr = datetime(curr.year, curr.month + 1, 1)
        return months

    def get_params(self, ignore_region=False):
        params = {}
        choice = self.date_choice.get()
        params['choice_mode'] = choice
        mes_map = {v: k for k, v in meses_dict.items()}
        
        # --- CONFIGURACIÓN DEL GAS ---
        selected_gas_key = self.gas_combo.get()
        gas_conf = GAS_CONFIG[selected_gas_key]
        params['gas_config'] = gas_conf
        gas_short = gas_conf['nombre_corto']
        
        # Generamos nombres de salida dinámicos según el gas
        def get_names(prefix_date):
            return {
                'output_name': f"Datos_{gas_short}_{prefix_date}",
                'output_name_blh': f"Datos_BLH_{prefix_date}"
            }

        if choice == "mes":
            mn = self.mes_combo.get(); m = mes_map[mn]; y = int(self.ano_spin.get())
            _, last = calendar.monthrange(y, m)
            params.update({'start_date': f"{y}-{m:02d}-01", 'end_date': f"{y}-{m:02d}-{last}", 'year': y, 'month': m, 'title_date': meses_es_lower[m], 'month_list': [(y, m)], 'title_suffix': f"{mn} {y}"})
            names = get_names(f"{mn}_{y}")
            params['output_name'] = f"{m:02d}_{names['output_name']}"
            params['output_name_blh'] = f"{m:02d}_{names['output_name_blh']}"
            
        elif choice == "dia":
            f = self.dia_cal.get_date().strftime('%Y-%m-%d')
            params.update({'start_date': f, 'end_date': f, 'year': int(f[:4]), 'title_date': f, 'title_suffix': f"Dia {f}"})
            params['output_name'] = f"Datos_{gas_short}_Dia_{f}"

        elif choice == "rango":
            s, e = self.rango_cal1.get_date(), self.rango_cal2.get_date()
            if s > e: raise ValueError("Inicio > Fin")
            s_str, e_str = s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')
            params.update({'start_date': s_str, 'end_date': e_str, 'year': s.year, 'month': s.month, 'title_date': f"Rango {s_str} a {e_str}", 'title_suffix': f"Rango {s_str} a {e_str}"})
            params['output_name'] = f"Datos_{gas_short}_Rango_{s_str}_a_{e_str}"
            params['output_name_blh'] = f"Datos_BLH_Rango_{s_str}_a_{e_str}"

        elif choice == "anio":
            y = int(self.ano_spin_completo.get())
            params.update({'year': y, 'month_list': [(y, m) for m in range(1, 13)], 'title_suffix': f"Año {y}"})

        elif choice == "rango_meses":
            m1n = self.rango_mes_combo_ini.get(); m2n = self.rango_mes_combo_fin.get()
            y1, y2 = int(self.rango_ano_spin_ini.get()), int(self.rango_ano_spin_fin.get())
            params.update({'month_list': self._get_month_list(y1, mes_map[m1n], y2, mes_map[m2n]), 'year': y1, 'title_suffix': f"Rango {m1n} {y1} - {m2n} {y2}"})

        # Región
        if not ignore_region:
            if self.region_mode.get() == "list":
                reg = self.region_combo.get()
                if not reg: raise ValueError("Seleccione región")
                params['aoi_path'] = BASE_GEOJSON_PATH / f"{reg}.geojson"
            else:
                try:
                    nm = self.manual_name_var.get().strip().replace(" ", "_")
                    if not nm: raise ValueError("Nombre manual requerido")
                    bbox_g = box(self.min_lon.get(), self.min_lat.get(), self.max_lon.get(), self.max_lat.get())
                    out = BASE_GEOJSON_PATH / f"Manual_{nm}.geojson"
                    gpd.GeoDataFrame({'geometry': [bbox_g]}, crs="EPSG:4326").to_file(out, driver="GeoJSON")
                    params['aoi_path'] = out
                except Exception as e: raise ValueError(f"Error manual: {e}")
        else: params['aoi_path'] = None

        params.update({"estadisticas": self.do_stats.get(), "generar_mapas": self.do_maps.get(), "do_zip": self.do_zip.get(),
            "show_plots": self.show_plots.get(), "palette": self.palette_combo.get(), "fig_queue": self.fig_queue,
            "formato_salida": self.formato_salida_var.get(), "do_regrid": self.do_regrid.get(), "do_comparative_map": self.do_comparative_map.get()})

        # Transformación
        params['transform_method'] = None; params['descargar_blh'] = False; params['shape_factor_pbl'] = None
        if self.do_transform.get():
            sel = self.transform_method_combo.get()
            if "Petetin" in sel: params['transform_method'] = "petetin"; params['descargar_blh'] = True
            elif "Savanets" in sel: params['transform_method'] = "savanets"
            elif "Valor escrito" in sel:
                val = simpledialog.askfloat("Altura H", "Ingrese valor H (metros):", parent=self.root, minvalue=1.0, maxvalue=50000.0)
                if val is None: return None
                params['transform_value'] = val
                params['transform_method'] = "custom"
            
            # Chequeo de Shape Factor para gases de troposfera gruesa
            if gas_conf["comportamiento_vertical"] == "Thick_troposphere":
                # Idealmente pediríamos esto por UI, pero por ahora lo dejamos default o hardcodeado según instrucciones
                # Opcional: Pedir al usuario si es modo experto
                # val_sf = simpledialog.askfloat("Shape Factor", f"Ingrese factor fracción PBL para {gas_short} (0-1):\nDefault: 1.0", parent=self.root, minvalue=0.0, maxvalue=1.0)
                # params['shape_factor_pbl'] = val_sf if val_sf else 1.0
                params['shape_factor_pbl'] = 1.0 # Default como solicitado

        return params

    def _execute_process_flow(self, params, cancel_event):
        gas_short = params['gas_config']['nombre_corto']
        if params.get('choice_mode') in ['anio', 'rango_meses']:
            print(f"🚀 INICIO PROCESO MULTI-MES ({gas_short})")
            no2_paths, blh_paths, conc_paths = [], [], []
            for y, m in params['month_list']:
                if cancel_event.is_set(): break
                p = params.copy()
                mn = meses_es_lower[m]; _, l = calendar.monthrange(y, m)
                p.update({'start_date': f"{y}-{m:02d}-01", 'end_date': f"{y}-{m:02d}-{l}", 'year': y, 'month': m, 'title_date': mn})
                p['output_name'] = f"{m:02d}_Datos_{gas_short}_{mn}_{y}"
                p['output_name_blh'] = f"{m:02d}_Datos_BLH_{mn}_{y}" if p['descargar_blh'] else f"Datos_Trans_{p.get('transform_method')}"
                
                n_p, b_p, c_p = run_processing(p, cancel_event)
                if n_p: no2_paths.append(n_p)
                if b_p: blh_paths.append(b_p)
                if c_p: conc_paths.append(c_p)
            
            if not cancel_event.is_set() and params['do_comparative_map']:
                q = params['fig_queue'] if params['show_plots'] else None
                if no2_paths: 
                    f = generar_mapa_comparativo(no2_paths, params['aoi_path'], f"{gas_short} (Columna)", "(mol/m²)", params['palette'], params['title_suffix'], params['show_plots'])
                    if f and q: q.put(f)
                if conc_paths and params.get('transform_method'):
                    f = generar_mapa_comparativo(conc_paths, params['aoi_path'], f"Concentración {gas_short}", "(ppb)", 'inferno', params['title_suffix'], params['show_plots'])
                    if f and q: q.put(f)
        else:
            run_processing(params, cancel_event)

    def processing_finished(self):
        self.toggle_buttons(True)
        print("\n✅ TAREA FINALIZADA" if not self.cancel_event.is_set() else "\n🛑 DETENIDO")
        self.cancel_event.clear()

    def check_fig_queue(self):
        try:
            fig = self.fig_queue.get_nowait()
            self.display_figure(fig)
        except queue.Empty: pass
        self.root.after(100, self.check_fig_queue)

    def display_figure(self, fig):
        win = tk.Toplevel(self.root); win.title("Visor")
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw()
        NavigationToolbar2Tk(canvas, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_console(self):
        self.console.config(state='normal'); self.console.delete('1.0', tk.END); self.console.config(state='disabled')

    class TextRedirector:
        def __init__(self, app): self.app = app
        def write(self, s):
            try:
                self.app.console.config(state='normal'); self.app.console.insert(tk.END, s); self.app.console.see(tk.END); self.app.console.config(state='disabled')
            except: pass
        def flush(self): pass

if __name__ == "__main__":
    if not BASE_GEOJSON_PATH.is_dir():
        messagebox.showerror("Error", f"Falta carpeta 'Regiones' en:\n{SCRIPT_DIR}")
        sys.exit(1)
    
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    root = tk.Tk()
    GeoApp(root, get_available_regions())
    root.mainloop()
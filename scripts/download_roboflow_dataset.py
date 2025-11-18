#!/usr/bin/env python3
"""
Script para descargar automáticamente datasets de Roboflow.

Este script facilita la descarga de datasets de baloncesto desde Roboflow Universe
usando la API de Roboflow. Soporta múltiples datasets y los organiza automáticamente.

Instalación requerida:
    pip install roboflow

Uso básico:
    python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY

Uso avanzado:
    # Descargar dataset específico
    python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --workspace roboflow-100 --project basketball-detection --version 1

    # Descargar múltiples datasets recomendados
    python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --download-all

Obtener API Key:
    1. Crea una cuenta gratis en: https://roboflow.com/
    2. Ve a Settings -> API Keys
    3. Copia tu Private API Key
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("❌ Error: El paquete 'roboflow' no está instalado.")
    print("\n📦 Instala con:")
    print("   pip install roboflow")
    sys.exit(1)


# Datasets recomendados de baloncesto en Roboflow Universe
# Nota: Estos son datasets públicos verificados como accesibles
RECOMMENDED_DATASETS = [
    {
        'name': 'Basketball Detection (Roboflow 100)',
        'workspace': 'roboflow-100',
        'project': 'basketball-detection',
        'version': 1,
        'description': 'Dataset oficial de Roboflow 100 - Alta calidad, bien anotado'
    },
]

# Datasets adicionales que puedes explorar en Roboflow Universe:
# - https://universe.roboflow.com/search?q=basketball
# - https://universe.roboflow.com/search?q=ball%20detection
#
# Para agregar un dataset personalizado, añádelo a RECOMMENDED_DATASETS siguiendo el formato:
# {
#     'name': 'Nombre del Dataset',
#     'workspace': 'nombre-workspace',
#     'project': 'nombre-proyecto',
#     'version': 1,
#     'description': 'Descripción breve'
# }


def download_dataset(api_key, workspace, project, version, output_dir='data/basketball_training'):
    """
    Descarga un dataset desde Roboflow.

    Args:
        api_key: API key de Roboflow
        workspace: Nombre del workspace
        project: Nombre del proyecto
        version: Versión del dataset
        output_dir: Directorio donde guardar el dataset

    Returns:
        Path al directorio del dataset descargado
    """
    try:
        print(f"\n📥 Descargando {workspace}/{project} v{version}...")

        # Inicializar Roboflow
        rf = Roboflow(api_key=api_key)

        # Obtener el proyecto
        project_obj = rf.workspace(workspace).project(project)

        # Obtener la versión específica
        version_obj = project_obj.version(version)

        # Crear directorio de salida
        dataset_path = os.path.join(output_dir, f"{project}_v{version}")
        os.makedirs(dataset_path, exist_ok=True)

        # Descargar en formato YOLOv8
        print("   Descargando en formato YOLOv8...")
        dataset = version_obj.download(
            model_format="yolov8",
            location=dataset_path,
            overwrite=False
        )

        print(f"   ✓ Dataset descargado: {dataset_path}")

        # Mostrar estadísticas
        train_images = os.path.join(dataset_path, 'train', 'images')
        valid_images = os.path.join(dataset_path, 'valid', 'images')

        if os.path.exists(train_images):
            train_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png'))])
            print(f"   📊 Imágenes de entrenamiento: {train_count}")

        if os.path.exists(valid_images):
            valid_count = len([f for f in os.listdir(valid_images) if f.endswith(('.jpg', '.png'))])
            print(f"   📊 Imágenes de validación: {valid_count}")

        return dataset_path

    except Exception as e:
        print(f"   ❌ Error descargando dataset: {str(e)}")
        print(f"\n💡 Verifica que el dataset existe en:")
        print(f"   https://universe.roboflow.com/{workspace}/{project}")
        return None


def download_recommended_datasets(api_key, output_dir='data/basketball_training'):
    """
    Descarga todos los datasets recomendados.

    Args:
        api_key: API key de Roboflow
        output_dir: Directorio donde guardar los datasets

    Returns:
        Lista de paths a los datasets descargados
    """
    print("\n" + "=" * 70)
    print("Descargando Datasets Recomendados de Baloncesto")
    print("=" * 70)

    downloaded = []

    for idx, dataset_info in enumerate(RECOMMENDED_DATASETS, 1):
        print(f"\n[{idx}/{len(RECOMMENDED_DATASETS)}] {dataset_info['name']}")
        print(f"    {dataset_info['description']}")

        dataset_path = download_dataset(
            api_key=api_key,
            workspace=dataset_info['workspace'],
            project=dataset_info['project'],
            version=dataset_info['version'],
            output_dir=output_dir
        )

        if dataset_path:
            downloaded.append(dataset_path)

    return downloaded


def list_available_datasets():
    """Muestra los datasets recomendados disponibles."""
    print("\n" + "=" * 70)
    print("Datasets Recomendados de Baloncesto")
    print("=" * 70)

    for idx, dataset_info in enumerate(RECOMMENDED_DATASETS, 1):
        print(f"\n{idx}. {dataset_info['name']}")
        print(f"   Workspace: {dataset_info['workspace']}")
        print(f"   Project: {dataset_info['project']}")
        print(f"   Version: {dataset_info['version']}")
        print(f"   Descripción: {dataset_info['description']}")
        print(f"   URL: https://universe.roboflow.com/{dataset_info['workspace']}/{dataset_info['project']}")

    print("\n" + "=" * 70)
    print("\nPara descargar todos los datasets:")
    print("python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --download-all")


def main():
    parser = argparse.ArgumentParser(
        description='Descarga datasets de baloncesto desde Roboflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Listar datasets recomendados
  python scripts/download_roboflow_dataset.py --list

  # Descargar todos los datasets recomendados
  python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY --download-all

  # Descargar dataset específico
  python scripts/download_roboflow_dataset.py --api-key YOUR_API_KEY \\
      --workspace roboflow-100 \\
      --project basketball-detection \\
      --version 1

Para obtener tu API key:
  1. Crea cuenta en https://roboflow.com/
  2. Ve a Settings -> API Keys
  3. Copia tu Private API Key
        """
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Tu API key de Roboflow (obtener en https://roboflow.com/)'
    )

    parser.add_argument(
        '--workspace',
        type=str,
        help='Nombre del workspace en Roboflow'
    )

    parser.add_argument(
        '--project',
        type=str,
        help='Nombre del proyecto en Roboflow'
    )

    parser.add_argument(
        '--version',
        type=int,
        default=1,
        help='Versión del dataset (default: 1)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/basketball_training',
        help='Directorio de salida (default: data/basketball_training)'
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Descargar todos los datasets recomendados'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='Listar datasets recomendados sin descargar'
    )

    args = parser.parse_args()

    # Mostrar lista de datasets
    if args.list:
        list_available_datasets()
        return 0

    # Verificar API key
    if not args.api_key:
        print("❌ Error: Se requiere --api-key para descargar datasets")
        print("\n📖 Obtén tu API key en: https://roboflow.com/")
        print("   Settings -> API Keys -> Private API Key")
        print("\nO usa --list para ver datasets disponibles")
        return 1

    # Descargar todos los datasets recomendados
    if args.download_all:
        downloaded = download_recommended_datasets(args.api_key, args.output_dir)

        print("\n" + "=" * 70)
        print(f"✅ Descarga completa: {len(downloaded)}/{len(RECOMMENDED_DATASETS)} datasets")
        print("=" * 70)

        if downloaded:
            print("\n📁 Datasets descargados en:")
            for path in downloaded:
                print(f"   - {path}")

            print("\n🚀 Siguiente paso: Entrenar el modelo")
            print("   python scripts/train_basketball_detector_simple.py")

        return 0

    # Descargar dataset específico
    if not args.workspace or not args.project:
        print("❌ Error: Se requiere --workspace y --project para descargar un dataset específico")
        print("\n💡 Usa --list para ver datasets recomendados")
        print("💡 Usa --download-all para descargar todos los recomendados")
        return 1

    dataset_path = download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=args.output_dir
    )

    if dataset_path:
        print("\n" + "=" * 70)
        print("✅ Descarga completa")
        print("=" * 70)
        print(f"\n📁 Dataset: {dataset_path}")
        print("\n🚀 Siguiente paso: Entrenar el modelo")
        print("   python scripts/train_basketball_detector_simple.py")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

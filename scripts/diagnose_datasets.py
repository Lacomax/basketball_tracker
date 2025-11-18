#!/usr/bin/env python3
"""
Script de diagnóstico para verificar estructura de datasets descargados.
Ayuda a identificar por qué el script de entrenamiento no encuentra datasets.
"""

import os
import sys
from pathlib import Path


def diagnose_directory(base_dir='data/basketball_training'):
    """Diagnostica la estructura de directorios de datasets."""

    print("=" * 70)
    print("DIAGNÓSTICO DE DATASETS")
    print("=" * 70)
    print(f"\nDirectorio base: {base_dir}")

    if not os.path.exists(base_dir):
        print(f"\n❌ El directorio no existe: {base_dir}")
        print("\nCrea el directorio con:")
        print(f"  mkdir -p {base_dir}  # Linux/Mac")
        print(f"  New-Item -ItemType Directory -Force -Path {base_dir}  # PowerShell")
        return

    print(f"✓ Directorio existe\n")

    # Listar contenido
    items = os.listdir(base_dir)
    if not items:
        print("❌ El directorio está vacío")
        print("\nDescarga datasets con:")
        print("  python scripts/download_roboflow_dataset.py --api-key YOUR_KEY --download-all")
        return

    print(f"📁 Encontrados {len(items)} item(s):\n")

    # Analizar cada item
    datasets_found = []

    for item in items:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            print(f"\n📂 {item}/")
            print("-" * 70)

            # Verificar estructura YOLO
            data_yaml = os.path.join(item_path, 'data.yaml')
            train_dir = os.path.join(item_path, 'train')
            train_images = os.path.join(item_path, 'train', 'images')
            train_labels = os.path.join(item_path, 'train', 'labels')
            valid_dir = os.path.join(item_path, 'valid')
            valid_images = os.path.join(item_path, 'valid', 'images')
            valid_labels = os.path.join(item_path, 'valid', 'labels')

            checks = {
                'data.yaml': os.path.exists(data_yaml),
                'train/': os.path.exists(train_dir),
                'train/images/': os.path.exists(train_images),
                'train/labels/': os.path.exists(train_labels),
                'valid/': os.path.exists(valid_dir),
                'valid/images/': os.path.exists(valid_images),
                'valid/labels/': os.path.exists(valid_labels),
            }

            for check_name, exists in checks.items():
                status = "✓" if exists else "✗"
                color = "green" if exists else "red"
                print(f"  {status} {check_name}")

                # Contar archivos si existe
                if exists and 'images' in check_name:
                    path = train_images if 'train' in check_name else valid_images
                    try:
                        count = len([f for f in os.listdir(path)
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"      → {count} imágenes")
                    except:
                        pass

                if exists and 'labels' in check_name:
                    path = train_labels if 'train' in check_name else valid_labels
                    try:
                        count = len([f for f in os.listdir(path)
                                   if f.endswith('.txt')])
                        print(f"      → {count} archivos de etiquetas")
                    except:
                        pass

            # Verificar si es válido
            is_valid = (os.path.exists(data_yaml) or os.path.exists(train_images))

            if is_valid:
                datasets_found.append(item_path)
                print(f"\n  ✅ VÁLIDO - Dataset YOLO detectado")
            else:
                print(f"\n  ⚠️  NO VÁLIDO - Estructura incorrecta")

                # Sugerencias
                print("\n  💡 Estructura esperada:")
                print("     dataset/")
                print("     ├── data.yaml")
                print("     ├── train/")
                print("     │   ├── images/")
                print("     │   └── labels/")
                print("     └── valid/")
                print("         ├── images/")
                print("         └── labels/")

        else:
            print(f"\n📄 {item} (archivo, ignorado)")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    if datasets_found:
        print(f"\n✅ Encontrados {len(datasets_found)} dataset(s) válido(s):")
        for ds in datasets_found:
            print(f"   - {Path(ds).name}")

        print("\n🚀 Puedes entrenar con:")
        print("   python scripts/train_basketball_detector_simple.py")
    else:
        print("\n❌ No se encontraron datasets válidos")
        print("\n📥 Opciones:")
        print("   1. Descarga datasets de Roboflow:")
        print("      python scripts/download_roboflow_dataset.py --api-key YOUR_KEY --download-all")
        print("\n   2. Verifica la estructura de los directorios arriba")
        print("\n   3. Si descargaste manualmente, asegúrate de:")
        print("      - Extraer en: data/basketball_training/nombre_dataset/")
        print("      - Formato: YOLOv8 (no Pascal VOC, COCO, etc.)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnostica estructura de datasets')
    parser.add_argument(
        '--dir',
        type=str,
        default='data/basketball_training',
        help='Directorio a diagnosticar'
    )

    args = parser.parse_args()

    diagnose_directory(args.dir)

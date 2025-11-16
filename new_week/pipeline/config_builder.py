#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfigBuilder class for YOLO inference configuration generation.

Extracted from PanoramaWithVirtualCamera to improve modularity and testability.
"""

import os
import logging

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """Handles creation and validation of YOLO inference configuration files."""

    def __init__(self):
        """Initialize ConfigBuilder."""
        pass

    def create_inference_config(self, output_path="config_infer.txt"):
        """Создание конфига для YOLO (только если файл отсутствует или некорректный)."""

        # Список обязательных полей для валидации
        required_fields = [
            'gpu-id',
            'model-engine-file',
            'batch-size',
            'network-mode',
            'num-detected-classes',
            'network-type',
            'output-blob-names',
            'pre-cluster-threshold',
            'nms-iou-threshold'
        ]

        # Проверка существования и валидности конфига
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    content = f.read()

                # Проверка, что файл не пустой
                if len(content.strip()) == 0:
                    logger.warning(f"⚠️ Конфиг {output_path} пустой, будет пересоздан")
                else:
                    # Проверка наличия всех обязательных полей
                    missing_fields = []
                    for field in required_fields:
                        if field not in content:
                            missing_fields.append(field)

                    if missing_fields:
                        logger.warning(f"⚠️ Конфиг {output_path} неполный (отсутствуют: {', '.join(missing_fields)}), будет пересоздан")
                    else:
                        # Проверка наличия секций [property] и [class-attrs-all]
                        if '[property]' not in content or '[class-attrs-all]' not in content:
                            logger.warning(f"⚠️ Конфиг {output_path} без необходимых секций, будет пересоздан")
                        else:
                            # Конфиг валидный, используем существующий
                            logger.info(f"✅ Используется существующий конфиг: {output_path}")
                            return output_path

            except Exception as e:
                logger.warning(f"⚠️ Ошибка чтения конфига {output_path}: {e}, будет пересоздан")

        # Создаём новый конфиг (если не существует или невалидный)
        config = """[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=../models/yolo11n_mixed_finetune_v9.engine
batch-size=6
network-mode=2
num-detected-classes=1
interval=1
gie-unique-id=1
process-mode=1
network-type=100
maintain-aspect-ratio=1
symmetric-padding=1
output-blob-names=output0
output-tensor-meta=1

[class-attrs-all]
pre-cluster-threshold=0.25
topk=100
nms-iou-threshold=0.45
"""
        with open(output_path, 'w') as f:
            f.write(config)
        with open("labels.txt", "w") as f:
            f.write("ball\n")
        logger.info(f"✅ Создан новый конфиг nvinfer: {output_path}")
        return output_path

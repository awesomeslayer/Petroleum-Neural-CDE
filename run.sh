#!/bin/bash

CMD=$1
GPU_ID=$2
# По умолчанию используем новый конфиг для месторождений
CONFIG_PATH=${3:-"./main/configs/petroleum_main.yaml"}

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  echo ">>> Using GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
else
  echo ">>> Using default GPU settings"
fi

case "$CMD" in
  main)
    echo ">>> Running Petroleum training with config: $CONFIG_PATH"
    # Запускаем новый скрипт как модуль
    python -m main.main_petroleum --config "$CONFIG_PATH"
    ;;

  agg|aggregate|plot|plots)
    # Временная заглушка, так как папка main/plots сейчас отсутствует
    echo ">>> Опция [$CMD] временно недоступна, так как папка 'main/plots' не найдена."
    echo ">>> Верните папку 'plots' в директорию 'main', чтобы использовать агрегацию и графики."
    ;;

  *)
    echo "Usage: $0 {main} [GPU_ID] [CONFIG_PATH]"
    echo "Example:"
    echo "  $0 main 0 ./main/configs/petroleum_main.yaml"
    exit 1
    ;;
esac
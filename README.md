# ThermoPINN

<div id="stack-badges">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
    </a>
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
    </a>
    <a href="https://deepxde.readthedocs.io/">
        <img src="https://img.shields.io/badge/DeepXDE-4B8BBE?style=for-the-badge&logo=deepxde&logoColor=white" alt="DeepXDE"/>
    </a>
    <a href="https://isocpp.org/">
        <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++"/>
    </a>
    <a href="https://www.openmp.org/">
        <img src="https://img.shields.io/badge/OpenMP-2A8E3C?style=for-the-badge&logo=openmp&logoColor=white" alt="OpenMP"/>
    </a>
    <a href="https://www.docker.com/">
        <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
    </a>
    <a href="https://wandb.ai/">
        <img src="https://img.shields.io/badge/Weights_&_Biases-FFCC00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="WandB"/>
    </a>
    <a href="https://cursor.sh/">
        <img src="https://img.shields.io/badge/Cursor-000000?style=for-the-badge&logo=cursor&logoColor=white" alt="Cursor AI Agent"/>
    </a>
</div>

-----

Моделирование температуры для задач лазерной медицины с помощью физически-информированных нейронных сетей.

Репозиторий включает:

- **C++ ADI** генератор синтетических данных;
- **PINN** на PyTorch/TensorFlow для **прямой** задачи (предсказание температуры) и **обратной** задачи (определение коэффициента теплопроводности);
- поддержку **Docker** для воспроизводимого окружения;
- примеры использования **AI-агента Cursor** при разработке.

## Структура репозитория

```
├── synthetic_data/          # классическое решение: C++ ADI солвер + эталонные выходы
│   ├── src/                 # исходники солвера
│   ├── data/                # синтетические данные
│   ├── scripts/             # визуализация
│   └── Makefile
├── pinn/                    # нейросетевое решение
│   ├── data/                # загрузка данных
│   ├── physics/             # домен, источник тепла, PDE
│   ├── models/              # FNN / MsFFN / STMsFFN + hard constraints
│   ├── problems/            # прямая и обратная задачи
│   ├── training/            # Trainer + логирование
│   └── visualization/
├── configs/                 # YAML конфиги экспериментов
├── scripts/                 # точки входа (CLI)
├── docker/                  # Dockerfile и зависимости
├── requirements.txt
└── README.md
```

## Установка и запуск

**Требования:** Python 3.10+, C++ компилятор с OpenMP (для синтетических данных), Docker (опционально).

### Локальная установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

**Настройка бэкенда DeepXDE**

- **FNN (по умолчанию):** PyTorch — переменная `DDE_BACKEND=pytorch` (установлена в скриптах по умолчанию).
- **MsFFN / STMsFFN:** требуется TensorFlow 1.x compatibility mode:

```bash
set DDE_BACKEND=tensorflow.compat.v1   # Windows CMD
export DDE_BACKEND=tensorflow.compat.v1  # Linux/macOS
```

### Запуск с Docker

Соберите образ:

```bash
docker build -f docker/Dockerfile -t thermopinn .
```

Запустите обучение (с GPU, если доступен):

```bash
docker run --gpus all -v $(pwd)/data:/app/data thermopinn python scripts/train_forward.py --config configs/forward_fnn.yaml
```

## Генерация синтетических данных (C++)

```bash
cd synthetic_data
make
cd build
../build/heat_diffusion_agar
```

Интерактивный ввод: скорость сканирования `v` [mm/s], период импульсов [s], энергия на импульс [J].  
Результаты записываются в `synthetic_data/data/` (`out_Tt.txt`, `out_Txz.txt`, …).

Визуализация эталонных полей:

```bash
python synthetic_data/scripts/visualize_results.py
```

## Прямая задача (предсказание температуры)

PINN решает уравнение теплопроводности с адиабатическими границами (и опциональными точками датчиков).

```bash
python scripts/train_forward.py --config configs/forward_fnn.yaml --plot
```

Сеть с Фурье-признаками (TensorFlow бэкенд):

```bash
set DDE_BACKEND=tensorflow.compat.v1
python scripts/train_forward.py --config configs/forward_msffn.yaml
```

## Обратная задача (определение коэффициента теплопроводности)

Неизвестный коэффициент `k` параметризуется как `log_k`, с `k = exp(log_k) > 0`. Обучение использует температурные привязки на глубинах **z = 0, 1, 3 мм** из `out_Tt.txt`.

```bash
python scripts/train_inverse.py --config configs/inverse_fnn.yaml --plot
```

После обучения скрипт выводит найденное значение `k` и сохраняет модель в `outputs/inverse_fnn/`.

## Конфигурация

YAML файлы в `configs/` управляют физикой, архитектурой сети, hard constraints, количеством коллокационных точек и фазами обучения (Adam / L-BFGS).

| Ключ | Описание |
|------|----------|
| `network.architecture` | `fnn`, `msffn`, `stmsffn` |
| `network.hard_constraint` | `ramp_softplus`, `linear_ramp`, `none` |
| `training.phases` | Список `{optimizer, lr, iterations}` |
| `wandb.enabled` | Включить/выключить логирование в Weights & Biases |


## Использование AI-агентов

Проект разрабатывался с активным применением **Cursor** — AI-агента для написания кода. Cursor помог в:

- генерации шаблонов архитектур PINN;
- создании Dockerfile и скриптов сборки;
- рефакторинге загрузчиков данных и визуализации;
- отладке функций потерь и hard constraints.

## Citation & license

Academic research project. 
```


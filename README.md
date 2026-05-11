# UGV Swarm Expert — MA-GAIL для керування роєм наземних роботів

Дипломний проєкт. Система навчання рою наземних мобільних роботів (UGV) утримувати задану формацію під час руху за лідером з використанням алгоритму **Multi-Agent GAIL (MA-GAIL)**.

---

## Загальний опис

Проєкт реалізує підхід **Imitation Learning** (навчання на основі демонстрацій) для керування роєм роботів TurtleBot 3 Waffle Pi у симуляторі Gazebo 11 (ROS 2 Humble).

Рій складається з одного **лідера** та двох або більше **ведених** агентів. Ведені роботи навчаються утримувати формацію (колона або клин) відносно лідера, уникаючи перешкод — без явно заданої функції нагороди, лише на основі експертних демонстрацій.

**Стек технологій:**
- Ubuntu 22.04 + ROS 2 Humble
- Gazebo 11 (класичний)
- Python 3.10, PyTorch
- TurtleBot 3 Waffle Pi (LiDAR, Odometry, IMU)

---

## Архітектура

```
ugv_swarm_expert/
├── data/           # Збір та підготовка експертних даних
├── env/            # ROS 2 середовище (StateProcessor, SafetySupervisor)
├── models/         # ActorNetwork, CriticNetwork, DiscriminatorNetwork
├── training/       # MA-GAIL тренувальний цикл (offline + online)
├── inference/      # Вузол інфлюенсу для задеплоєної моделі
├── navigation/     # LeaderNavigator (waypoint / manual / nav2)
└── evaluation/     # Метрики оцінки якості формації
```

**Нейромережа (Actor):** гібридна архітектура — 1D-CNN для LiDAR-даних + Dense-гілка для кінематики → MLP (256→128→64) з Layer Normalization.

**Дискримінатор:** Multi-Head Attention по всіх агентах → Global Max Pooling → MLP → Sigmoid.

**Метрики оцінки:**
- `E_f` — середня похибка формації (см)
- `SR` — відсоток успішно завершених місій
- `S_ω` — плавність керування
- `T_rec` — час відновлення формації після обходу перешкоди

---

## Встановлення

```bash
# Клонувати репозиторій у ROS 2 workspace
cd ~/ros2_ws/src
git clone <repo_url> diploma-project

# Встановити Python-залежності
pip install -r diploma-project/requirements.txt

# Зібрати пакет
cd ~/ros2_ws
colcon build --packages-select ugv_swarm_expert
source install/setup.bash
```

---

## Швидкий старт

### 1. Підготовка експертних даних

```bash
# Запустити симуляцію та вузол збору даних
ros2 launch ugv_swarm_expert swarm_simulation.launch.py

# В окремому терміналі — збір демонстрацій
ros2 run ugv_swarm_expert expert_data_collector

# Перетворити CSV у тензори для навчання
python ugv_swarm_expert/scripts/prepare_expert_data.py \
    --input datasets/dataset.csv \
    --output datasets/expert_tensors.pt
```

### 2. Офлайн-навчання

```bash
python ugv_swarm_expert/scripts/offline_train.py \
    --expert-data datasets/expert_tensors.pt \
    --epochs 100 \
    --batch-size 512 \
    --checkpoint-dir checkpoints/offline
```

### 3. Запуск інференсу в симуляції

```bash
# Запустити всю систему: симуляція + лідер + ведені агенти
ros2 launch ugv_swarm_expert swarm_inference.launch.py \
    model_path:=checkpoints/offline/actor_epfinal.pth \
    formation_type:=column \
    formation_distance:=0.7 \
    follower_names:=tb3_1,tb3_2
```

#### Клин замість колони

```bash
ros2 launch ugv_swarm_expert swarm_inference.launch.py \
    formation_type:=wedge \
    formation_angle_deg:=45.0 \
    formation_distance:=0.8
```

#### Без симуляції (тільки вузли керування)

```bash
ros2 launch ugv_swarm_expert swarm_inference.launch.py \
    start_simulation:=false \
    model_path:=checkpoints/offline/actor_epfinal.pth
```

### 4. Оцінка якості моделі

```bash
ros2 run ugv_swarm_expert eval_runner \
    --actor checkpoints/offline/actor_epfinal.pth \
    --num-agents 3 \
    --output results/eval_report.json
```

### 5. Запуск у світі з конусами-перешкодами

```bash
ros2 launch ugv_swarm_expert cones_world.launch.py use_gui:=true
```

---

## Тести

```bash
cd ugv_swarm_expert
pytest test/ -v
```

---

## Параметри launch-файлу `swarm_inference.launch.py`

| Параметр | За замовчуванням | Опис |
|---|---|---|
| `model_path` | `checkpoints/offline/actor_epfinal.pth` | Шлях до `.pth` чекпоінту актора |
| `formation_type` | `column` | Тип формації: `column` або `wedge` |
| `formation_distance` | `0.7` | Відстань між роботами (м) |
| `follower_names` | `tb3_1,tb3_2` | Імена ведених агентів |
| `leader_mode` | `waypoint` | Режим лідера: `manual`, `waypoint`, `nav2` |
| `use_gui` | `true` | Запускати Gazebo GUI |
| `device` | `auto` | PyTorch пристрій: `cpu`, `cuda:0`, `auto` |
| `start_simulation` | `true` | Запускати Gazebo разом із системою |

---

## Структура даних (CSV)

| Поле | Опис |
|---|---|
| `timestamp`, `agent_id` | Часова мітка та ідентифікатор агента |
| `odom_x`, `odom_y`, `odom_theta` | Глобальне положення з одометрії |
| `rel_dx`, `rel_dy` | Відносні координати до лідера |
| `lidar_sector_1..36` | Мінімуми по 36 секторах LiDAR (по 10°) |
| `action_v`, `action_w` | Лінійна та кутова швидкості (дії експерта) |

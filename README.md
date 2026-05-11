# Керування формацією рою БНМ за допомогою MA-GAIL

## Огляд проекту

Репозиторій містить реалізацію системи планування шляху та керування формацією рою безпілотних наземних машин (БНМ) на основі багатоагентного генеративно-змагального навчання з імітації (Multi-Agent Generative Adversarial Imitation Learning, MA-GAIL).

Система використовує децентралізовану нейромережеву політику для того, щоб рій диференційно-приводних БНМ (TurtleBot 3 Waffle Pi) автономно утримував геометричні формації ("Колона", "Клин/V-подібна") та обходив статичні й динамічні перешкоди. Підхід на основі імітаційного навчання дозволяє синтезувати складні стратегії керування з експертних демонстрацій без ручного проектування функції нагороди.

---

## Архітектура: централізоване навчання, децентралізоване виконання (CTDE)

Ключова архітектурна парадигма проекту — CTDE (Centralized Training, Decentralized Execution).

1. **Фаза навчання.** Централізований Дискримінатор є глобальним спостерігачем і має повний доступ до об'єднаного простору станів і дій усього рою. Це дозволяє оцінювати топологію рою, узгодженість формації та якість координації агентів.

2. **Фаза виконання (інференс).** Дискримінатор вимикається. Кожен БНМ запускає власну мережу-Генератор (Актор) автономно, приймаючи рішення виключно на основі локальних сенсорних даних: LiDAR, локальна одометрія, вектор до цільового слота формації.

---

## Структура пайплайну

```
[Збір даних] -> [Попередня обробка] -> [Feature Engineering]
     -> [Навчання (MA-GAIL / PPO)] -> [Інференс у ROS 2]
```

---

## 1. Збір експертних демонстрацій

Вузол `expert_data_collector` реалізує детермінований алгоритмічний експерт типу "лідер-послідовник". Він підписується на топіки одометрії та LiDAR кожного агента, публікує команди руху для фолловерів і записує CSV-файл на кожного фолловера.

### Простір станів і дій

Система працює на частоті 10 Гц (dt = 0.1 с). Для врахування інерції та запобігання ривкам використовується ковзне вікно з k=4 кадрів (0.4 с).

- Форма входу: `(Batch, 4, 41)` на одного агента. 41 ознака:
  - Кінематика (3): лінійна швидкість v, кутова швидкість omega, курс theta.
  - Формація (2): відносне зміщення [dx, dy] до цільового слота у локальній системі координат лідера.
  - Сприйняття (36): 360 сирих променів LiDAR агрегуються min-пулінгом у 36 секторів по 10 градусів.
- Форма виходу: `(Batch, 2)`. Лінійна швидкість v_cmd в [0, 0.22] м/с і кутова omega_cmd в [-2.84, 2.84] рад/с.

### Запуск збору даних

Спочатку запустіть симуляцію Gazebo (у окремому терміналі):

```bash
ros2 launch ugv_swarm_expert empty_world.launch.py
# або: ros2 launch ugv_swarm_expert cones_world.launch.py
```

Потім запустіть вузол збору даних:

```bash
ros2 run ugv_swarm_expert expert_data_collector --ros-args
```

Основні параметри вузла:

| Параметр | Значення за замовчуванням | Опис |
|---|---|---|
| `leader_name` | `leader` | Ім'я лідера |
| `follower_names` | `['tb3_1', 'tb3_2']` | Список фолловерів |
| `formation_distance` | `0.7` | Відстань між агентами у колоні, м |
| `formation_offsets` | — | Явні зміщення `dx,dy;dx,dy` (для клину: `-0.7,0.5;-0.7,-0.5`) |
| `output_dir` | `~/ugv_swarm_expert_data` | Директорія для CSV |
| `max_data_age_sec` | `0.5` | Поріг актуальності повідомлень. `0.0` — вимкнути |

Формат CSV-файлу:

```
time_step,pos_x,pos_y,yaw,rel_dist_lead,rel_ang_lead,lidar_s1,...,lidar_s36,target_v,target_w
```

---

## 2. Попередня обробка даних

Модуль `dataset_preprocessor` очищує сирі CSV-траєкторії та синхронізує всіх агентів до єдиної сітки 10 Гц.

```bash
ros2 run ugv_swarm_expert dataset_preprocessor \
  --agent-csv <agent>=<path_to_raw.csv> [...] \
  --output <path_to_clean_dataset.csv>
```

Що виконує препроцесор:

- Видаляє кадри з фізичними аномаліями Gazebo та наступне вікно стабілізації 0.5 с.
- Обрізає всі 36 секторів LiDAR до діапазону [0.12, 3.5] м.
- Інтерполює x, y, theta, v, omega на спільну часову сітку 10 Гц.
- Розгортає кут рискання перед інтерполяцією, щоб уникнути розривів на кордоні -pi/pi.
- Записує широкий CSV з одним стовпцем `time` та синхронізованими ознаками з префіксом агента.

---

## 3. Feature Engineering

Модуль `feature_engineer` перетворює очищений CSV на PyTorch-тензори для навчання актора.

```bash
ros2 run ugv_swarm_expert feature_engineer \
  --input <clean_dataset.csv> \
  --output <tensors.pt> \
  --leader <leader_name> \
  --followers <follower1> <follower2> \
  --target-offset <follower1>=<dx,dy> [...]
```

Що виконує модуль:

- Min-Max нормалізація всіх ознак до [0, 1] на основі апаратних обмежень.
- Дзеркальна аугментація: траєкторії відображаються горизонтально (інвертуються dy, theta, omega, реверсуються сектори LiDAR) — подвоює розмір датасету та усуває статистичне зміщення.
- Формування 4-кадрових ковзних вікон з паддінгом на першому кроці.

Використання як PyTorch Dataset:

```python
from ugv_swarm_expert.feature_engineer import UGVSwarmDataset

dataset = UGVSwarmDataset(
    "/path/to/clean_swarm_dataset.csv",
    leader_name="leader",
    follower_names=["tb3_1", "tb3_2"],
    target_offsets={"tb3_1": (-0.7, 0.0), "tb3_2": (-1.4, 0.0)},
)
state_sequence, action = dataset[0]
# state_sequence.shape == (4, 41)
# action.shape == (2,)
```

---

## 4. Архітектура моделей

### Генератор / Актор

Актор оптимізується алгоритмом PPO (Proximal Policy Optimization) і відображає простір станів на дії.

- LiDAR Encoder: 1D-CNN (16 фільтрів, ядро 3) -> ReLU -> MaxPool(2) -> Flatten -> Dense(32).
- Кінематична гілка: Flatten -> Dense(32).
- Fusion MLP: конкатенований 64-вимірний вектор -> Dense(256) -> LayerNorm -> ReLU -> Dense(128) -> LayerNorm -> ReLU -> Dense(64) -> LayerNorm -> ReLU.
- Вихід: Dense(2) з активацією Tanh. Дії семплюються з нормального розподілу з вихідним середнім та навченим параметром sigma.

### Дискримінатор

Дискримінатор оцінює, наскільки сгенеровані траєкторії відповідають експертним демонстраціям.

- Вхід: об'єднаний простір станів і дій `(Batch, N, 43)`.
- Кодування: локальний MLP (Dense(64, LeakyReLU)) на кожного агента.
- Агрегація: Multi-Head Attention + Global Max Pooling — топологічно інваріантне представлення рою.
- Оцінювач: Dense(256) -> LeakyReLU -> Dense(128) -> LeakyReLU -> Dense(64) -> LeakyReLU -> Dense(1, Sigmoid).

---

## 5. Навчання (PPO + MA-GAIL)

Навчання проводиться офлайн за наявності зібраних тензорів у форматі `.pt`.

### Підготовка середовища

```bash
cd /path/to/ros2_ws
colcon build --packages-select ugv_swarm_expert
source install/setup.zsh
```

### Запуск тренування

```bash
python ugv_swarm_expert/scripts/offline_train.py \
  --expert-data <tensors.pt> \
  --checkpoint-dir <checkpoints_dir>
```

або через ROS 2:

```bash
ros2 run ugv_swarm_expert ma_gail_train \
  --expert-data <tensors.pt> \
  --checkpoint-dir <checkpoints_dir>
```

### Принцип роботи навчання

Генератор не отримує винагороди від середовища. Натомість він максимізує сурогатну нагороду від Дискримінатора:

```
r(s, a) = -log(1 - D(s, a))
```

Дискримінатор оптимізується на BCE-втраті: відокремлює експертні дані (D ≈ 1) від траєкторій агента (D ≈ 0). PPO-оновлення застосовує відсічений сурогатний об'єктив, ентропійний бонус та MSE-критик.

Ключові гіперпараметри:

| Параметр | Значення |
|---|---|
| Оптимізатор | Adam |
| Learning rate | 3e-4 |
| Batch size | 128 |
| Discount factor gamma | 0.99 |
| Entropy coefficient | 0.01 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |

Кожна епоха включає: збір роллаутів, обчислення GAIL-нагороди, GAE, оновлення дискримінатора, оновлення Актора та Критика через PPO, логування в TensorBoard, збереження чекпоінту.

---

## 6. Запуск симуляції в Gazebo

```bash
# Порожній світ
ros2 launch ugv_swarm_expert empty_world.launch.py

# Світ з конусами (перешкоди)
ros2 launch ugv_swarm_expert cones_world.launch.py
```

### Параметри запуску симуляції

| Параметр | Значення за замовчуванням | Опис |
|---|---|---|
| `world_file` | `empty_world.sdf` | Шлях до SDF-файлу світу |
| `use_gui` | `true` | Запустити gzclient (вимкнути для headless) |
| `use_sim_time` | `true` | Використовувати /clock від Gazebo |

Симуляція автоматично спавнить трьох роботів: лідера (`leader`) на позиції (0, 0) та двох фолловерів (`tb3_1`, `tb3_2`) у колоні з кроком 0.7 м.

---

## 7. Інференс навченої моделі

Вузол `inference_node` запускає навчену політику Актора на одному фолловері. Він підписується на локальну одометрію та LiDAR, а також на `/leader/odom`, підтримує нормалізоване 4-кадрове вікно стану та публікує команди `/{robot_namespace}/cmd_vel` на частоті 10 Гц.

```bash
ros2 run ugv_swarm_expert inference_node \
  --ros-args -p robot_namespace:=<name> -p model_path:=<path/to/actor.pth> [параметри]
```

Для запуску повного рою використовуйте launch-файл:

```bash
ros2 launch ugv_swarm_expert swarm_inference.launch.py
```

Навігація лідера запускається окремо:

```bash
ros2 launch ugv_swarm_expert leader_navigation.launch.py
```

---

## 8. Оцінка якості моделі

Модуль `eval_metrics` оцінює навчений Актор у Gazebo та обчислює метрики SR, E_f, S_omega, T_rec.

```bash
ros2 run ugv_swarm_expert eval_metrics \
  --actor <path/to/actor.pth> [параметри]
```

Метрики:

| Метрика | Опис |
|---|---|
| SR (Success Rate) | Частка успішних епізодів |
| E_f (Formation Error) | Середня помилка формації, м |
| S_omega (Smoothness) | Плавність керуючих сигналів |
| T_rec (Recovery Time) | Час відновлення формації після контакту з перешкодою, с |

---

## 9. Результати

Порівняно з класичним Behaviour Cloning (BC), MA-GAIL демонструє вищу стійкість:

| Метрика | MA-GAIL | BC |
|---|---|---|
| SR (Success Rate) | 96.0% | ~72% |
| E_f (Formation Error) | 0.039 м | ~0.082 м |
| S_omega (Smoothness) | 0.38 | ~2.13 |
| T_rec (Recovery Time) | 1.5 с | — |

---

## 10. Встановлення та залежності

### Вимоги

- Ubuntu 22.04 LTS або 24.04 LTS
- ROS 2 Humble або Jazzy
- Python 3.10+
- PyTorch 2.x (CUDA 12.x рекомендовано)
- TurtleBot3 пакети для ROS 2

### Встановлення TurtleBot3

```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-description
export TURTLEBOT3_MODEL=waffle_pi
```

### Збірка пакету

```bash
cd /path/to/ros2_ws
colcon build --packages-select ugv_swarm_expert
source install/setup.zsh
```

### Python-залежності

```bash
pip install -r requirements.txt
```

---

## Структура репозиторію

```
ugv_swarm_expert/
    ugv_swarm_expert/          # Основний Python-пакет
        data/                  # StateProcessor, DatasetPreprocessor, FeatureEngineer
        env/                   # UGVSwarmEnv — обгортка Gym для ROS 2/Gazebo
        models/                # ActorNetwork, CriticNetwork, DiscriminatorNetwork
        training/              # MAGAILTrainer, PPORolloutBuffer, GAILReward, train.py
        inference/             # UGVInferenceNode
        navigation/            # LeaderNavigator
        evaluation/            # EvalMetrics
    scripts/
        offline_train.py       # Скрипт офлайн-тренування
        prepare_expert_data.py # Скрипт підготовки даних
    launch/
        swarm_simulation.launch.py   # Базовий запуск симуляції
        empty_world.launch.py        # Запуск у порожньому світі
        cones_world.launch.py        # Запуск у світі з перешкодами
        leader_navigation.launch.py  # Навігація лідера
        swarm_inference.launch.py    # Запуск інференсу рою
    worlds/
        empty_world.sdf
        cones_world.sdf
    datasets/
        dataset.csv
        expert_tensors.pt
    requirements.txt
```

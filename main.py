import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx

# --- Parametreler ---
WORLD_SIZE = 10  # NxN boyutunda dünya
NUM_DRONES = 4  # Drone sayısı
SENSOR_RANGE = 2  # Sensör yarıçapı
FIRE_CELLS = 5  # Başlangıçtaki yangın sayısı
MAX_BATTERY = 50  # Drone'ların maksimum batarya kapasitesi
MAX_EXTINGUISH = 10  # Drone'ların söndürücü kapasitesi
BASE_LOCATION = (0, 0)  # Üs noktası konumu
BATTERY_CONSUMPTION_PER_STEP = 1  # Her adımda harcanan batarya miktarı
DRONE_COLORS = ['blue', 'green', 'orange', 'purple']  # Her drone için farklı renkler
FIRE_SPREAD_PROBABILITY = 0.1  # Yangının yayılma ihtimali (daha yavaş yayılma)
BURN_TIME = 10  # Yangın hücresinin tamamen yanma süresi (adım sayısı)
LOG_FILE = "simulation_log.txt"  # Log dosyası adı
GIF_FILE = "simulation_output.gif"  # GIF dosyası adı

# --- Dünya Modeli ---
class World:
    def __init__(self, size, fire_cells):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0: Boş, 1: Yangın
        self.burn_time = np.zeros((size, size), dtype=int)  # Hücre başına kalan yanma süresi
        self.visited = np.zeros((size, size), dtype=bool)  # Ziyaret edilen hücreler
        self.burned_out = np.zeros((size, size), dtype=bool)  # Tamamen yanmış hücreler
        self.total_fire_cells = fire_cells
        self.extinguished_fire_cells = 0
        self._place_fires(fire_cells)

    def _place_fires(self, fire_cells):
        placed_fires = 0
        while placed_fires < fire_cells:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x, y] == 0:  # Eğer hücrede zaten yangın yoksa
                self.grid[x, y] = 1  # Yangını yerleştir
                self.burn_time[x, y] = BURN_TIME
                placed_fires += 1

    def mark_visited(self, x, y):
        self.visited[x, y] = True

    def is_visited(self, x, y):
        return self.visited[x, y]

    def spread_fire(self):
        """Yangını rastgele hücrelere yay."""
        new_grid = np.zeros((self.size, self.size), dtype=int)
        new_burn_time = np.zeros((self.size, self.size), dtype=int)

        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y] == 1 and not self.burned_out[x, y]:  # Eğer hücrede yangın varsa
                    # Yanma süresini azalt
                    self.burn_time[x, y] -= 1
                    if self.burn_time[x, y] <= 0:
                        # Hücre tamamen yandı
                        self.burned_out[x, y] = True
                        continue

                    # Yangını hücrede tut
                    new_grid[x, y] = 1
                    new_burn_time[x, y] = self.burn_time[x, y]

                    # Yangını yay (yalnızca sağ, sol, yukarı, aşağı yönlerinde)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size and new_grid[nx, ny] == 0 and not self.burned_out[nx, ny]:
                            if random.random() < FIRE_SPREAD_PROBABILITY:
                                new_grid[nx, ny] = 1
                                new_burn_time[nx, ny] = BURN_TIME
                                self.total_fire_cells += 1

        self.grid = new_grid
        self.burn_time = new_burn_time

    def extinguish_fire(self, x, y):
        """Yangını söndür ve kaydı tut."""
        if self.grid[x, y] == 1:
            self.grid[x, y] = 0
            self.extinguished_fire_cells += 1

    def get_burned_area(self):
        """Tamamen yanmış alanların sayısını döndür."""
        return np.sum(self.burned_out)

    def get_fire_statistics(self):
        """Yangın istatistiklerini döndür."""
        remaining_fires = np.sum(self.grid == 1)
        total_burned = self.get_burned_area()
        return {
            "total_fire_cells": self.total_fire_cells,
            "extinguished_fire_cells": self.extinguished_fire_cells,
            "remaining_fires": remaining_fires,
            "burned_out": total_burned
        }

# --- Drone Modeli ---
class Drone:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.battery = MAX_BATTERY
        self.extinguisher = MAX_EXTINGUISH
        self.returning_to_base = False
        self.target_fire = None
        self.extinguished_fires = 0

    def sense(self, world):
        """Sensör menzili içindeki yangınları algılar."""
        fires = []
        for i in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
            for j in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
                nx, ny = self.x + i, self.y + j
                if 0 <= nx < world.size and 0 <= ny < world.size and world.grid[nx, ny] == 1:
                    fires.append((nx, ny))
        return fires

    def move(self, dx, dy, world):
        """Drone'u belirli bir yöne hareket ettir."""
        if self.battery > 0:
            new_x = self.x + dx
            new_y = self.y + dy

            # Alan sınırları içinde kal
            if 0 <= new_x < world.size and 0 <= new_y < world.size:
                self.x = new_x
                self.y = new_y
                self.battery -= BATTERY_CONSUMPTION_PER_STEP
                world.mark_visited(self.x, self.y)

    def extinguish(self, world):
        """Drone bulunduğu hücrede yangını söndürür."""
        if world.grid[self.x, self.y] == 1 and self.extinguisher > 0:
            world.extinguish_fire(self.x, self.y)
            self.extinguisher -= 1
            self.extinguished_fires += 1

    def recharge_and_refill(self):
        """Drone bataryasını ve söndürücüsünü üs noktasında doldurur."""
        if self.x == BASE_LOCATION[0] and self.y == BASE_LOCATION[1]:
            self.battery = MAX_BATTERY
            self.extinguisher = MAX_EXTINGUISH
            self.returning_to_base = False

    def distance_to_base(self):
        """Drone'un üsse olan Manhattan mesafesi."""
        return abs(self.x - BASE_LOCATION[0]) + abs(self.y - BASE_LOCATION[1])

# --- MLAT-R Algoritması ---
def mlat_r(drone, world, all_drones, action_tree):
    """Drone için en iyi hareketi hesapla ve aksiyon ağacına ekle."""
    if drone.returning_to_base or drone.battery <= drone.distance_to_base() * BATTERY_CONSUMPTION_PER_STEP or drone.extinguisher == 0:
        # Üsse dön
        drone.returning_to_base = True
        dx = np.sign(BASE_LOCATION[0] - drone.x)
        dy = np.sign(BASE_LOCATION[1] - drone.y)
        action_tree.add_edge((drone.x, drone.y), (drone.x + dx, drone.y + dy))
        return dx, dy

    # Eğer drone'un hedef yangını varsa o yangına hareket et
    if drone.target_fire is not None and drone.target_fire in zip(*np.where(world.grid == 1)):
        target_x, target_y = drone.target_fire
        dx = np.sign(target_x - drone.x)
        dy = np.sign(target_y - drone.y)
        action_tree.add_edge((drone.x, drone.y), (drone.x + dx, drone.y + dy))
        return dx, dy

    # Yangın tespit et ve bir hedef belirle
    fires = drone.sense(world)
    if fires:
        # Yangına öncelik ver
        drone.target_fire = fires[0]
        dx = np.sign(fires[0][0] - drone.x)
        dy = np.sign(fires[0][1] - drone.y)
        action_tree.add_edge((drone.x, drone.y), (drone.x + dx, drone.y + dy))
        return dx, dy

    # Yeni alanları keşfet
    unvisited_cells = [(i, j) for i in range(world.size) for j in range(world.size) if not world.is_visited(i, j)]
    if unvisited_cells:
        closest_unvisited = min(unvisited_cells, key=lambda cell: abs(drone.x - cell[0]) + abs(drone.y - cell[1]))
        dx = np.sign(closest_unvisited[0] - drone.x)
        dy = np.sign(closest_unvisited[1] - drone.y)
        action_tree.add_edge((drone.x, drone.y), (drone.x + dx, drone.y + dy))
        return dx, dy

    # Eğer tüm alanlar ziyaret edildiyse rastgele hareket et
    dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    action_tree.add_edge((drone.x, drone.y), (drone.x + dx, drone.y + dy))
    return dx, dy

# --- Simülasyon ---
def simulate():
    world = World(WORLD_SIZE, FIRE_CELLS)
    drones = [Drone(i, random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1)) for i in range(NUM_DRONES)]
    action_tree = nx.DiGraph()

    fig, ax = plt.subplots()
    ax.set_xlim(-1, WORLD_SIZE)
    ax.set_ylim(-1, WORLD_SIZE)
    ax.set_xticks(range(WORLD_SIZE))
    ax.set_yticks(range(WORLD_SIZE))
    ax.grid(True)

    with open(LOG_FILE, "w") as log_file:
        log_file.write("Step,Active_Fires,Drone_ID,Extinguished_Fires\n")

        def update(frame):
            ax.clear()
            ax.set_xlim(-1, WORLD_SIZE)
            ax.set_ylim(-1, WORLD_SIZE)
            ax.set_xticks(range(WORLD_SIZE))
            ax.set_yticks(range(WORLD_SIZE))
            ax.grid(True)

            world.spread_fire()

            active_fires = np.sum(world.grid == 1)
            for drone in drones:
                if drone.battery > 0:
                    dx, dy = mlat_r(drone, world, drones, action_tree)
                    drone.move(dx, dy, world)
                    drone.extinguish(world)
                    drone.recharge_and_refill()

                # Log yaz
                log_file.write(f"{frame},{active_fires},{drone.id},{drone.extinguished_fires}\n")

            # Çizimleri güncelle
            fires_x, fires_y = np.where(world.grid == 1)
            ax.plot(fires_y, fires_x, 'rx', markersize=10, label='Yangın')
            for drone in drones:
                ax.plot(drone.y, drone.x, marker='o', color=DRONE_COLORS[drone.id % len(DRONE_COLORS)], label=f'Drone {drone.id}')

            # Üs noktasını çiz
            ax.plot(BASE_LOCATION[1], BASE_LOCATION[0], 'gs', markersize=12, label='Üs')
            ax.legend()

            # Yangınlar biterse simülasyonu sonlandır
            if active_fires == 0:
                print(f"Simülasyon tamamlandı! {frame} adımda bitti.")
                stats = world.get_fire_statistics()
                log_file.write(f"\nToplam Yangın Hücreleri: {stats['total_fire_cells']}\n")
                log_file.write(f"Söndürülen Yangınlar: {stats['extinguished_fire_cells']}\n")
                log_file.write(f"Sönmeyen Yangınlar: {stats['remaining_fires']}\n")
                log_file.write(f"Tamamen Yanmış Alanlar: {stats['burned_out']}\n")
                plt.close()

        ani = FuncAnimation(fig, update, frames=100, repeat=False, interval=500)
        ani.save(GIF_FILE, writer=PillowWriter(fps=10))
        print(f"Animasyon kaydedildi: {GIF_FILE}")

simulate()

import numpy as np
import pygame

# =============================
# Planet Class
# =============================
class Planet:
    def __init__(self, position, velocity, mass, radius, color=(255, 255, 255), name="Planet"):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.name = name

    def update_position(self, dt):
        self.position += self.velocity * dt


# =============================
# Simulation Class
# =============================
class GravitySimulator:
    def __init__(self, G=1.0):
        self.G = G
        self.planets = []

    def add_planet(self, planet):
        self.planets.append(planet)

    def compute_forces(self):
        n = len(self.planets)
        positions = np.array([p.position for p in self.planets])
        masses = np.array([p.mass for p in self.planets])
        accelerations = np.zeros_like(positions)

        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[j] - positions[i]
                r2 = np.dot(diff, diff)
                r = np.sqrt(r2)

                if r == 0:
                    continue

                force_dir = diff / r
                force_mag = self.G * masses[i] * masses[j] / r2

                accelerations[i] += force_mag * force_dir / masses[i]
                accelerations[j] -= force_mag * force_dir / masses[j]

        return accelerations

    def step(self, dt):
        accelerations = self.compute_forces()
        for i, p in enumerate(self.planets):
            p.velocity += accelerations[i] * dt
            p.update_position(dt)
        self.detect_collisions()

    def detect_collisions(self):
        n = len(self.planets)
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = self.planets[i], self.planets[j]
                dist = np.linalg.norm(pi.position - pj.position)
                if dist <= pi.radius + pj.radius:
                    print(f"Collision detected: {pi.name} <-> {pj.name}")


# =============================
# Pygame Visualization
# =============================
def run_simulation():
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Create simulation
    sim = GravitySimulator(G=10.0)

    # Example planets
    p1 = Planet(position=[-100, 0], velocity=[0, 0.6], mass=50, radius=8, color=(0, 200, 255), name="P1")
    p2 = Planet(position=[100, 0], velocity=[0, -0.6], mass=50, radius=8, color=(255, 100, 0), name="P2")
    p3 = Planet(position=[0, 200], velocity=[-0.5, 0], mass=20, radius=5, color=(0, 255, 100), name="P3")

    sim.add_planet(p1)
    sim.add_planet(p2)
    sim.add_planet(p3)

    # Scaling: world coords → screen coords
    scale = 1.0  # pixels per world unit
    offset = np.array([WIDTH // 2, HEIGHT // 2])

    running = True
    dt = 0.1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.step(dt)

        screen.fill((0, 0, 0))  # clear screen

        for planet in sim.planets:
            screen_pos = planet.position * scale + offset
            pygame.draw.circle(screen, planet.color, screen_pos.astype(int), planet.radius)

        pygame.display.flip()
        clock.tick(60)  # limit FPS

    pygame.quit()


if __name__ == "__main__":
    run_simulation()

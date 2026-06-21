"""Mode 1: the pygame viewer plus teaching input (proves within-life learning).

Dead-simple rendering: food are little green dots, hazards are red circles,
creatures are colored circles with a line showing heading. Click a creature to
select it; hold R to inject reward and P to inject punishment into the selected
creature's biochemistry -- that is the teaching channel.

A simple task is set up for you to teach: reward the creature for approaching
food, punish it for approaching the hazard. The HUD plots the selected creature's
food-eating rate over time so the improvement from teaching is visible, and a
frozen "control" creature (no plasticity) is tracked alongside so you can see the
taught creature pull ahead. Metrics are written to a CSV on exit so the curve is
plottable.
"""

from __future__ import annotations

import csv as csvmod

import numpy as np

from config import Config
from genome import Genome
from world import World


# colors
BG = (18, 18, 24)
FOOD_C = (90, 200, 90)
HAZARD_C = (200, 70, 70)
SELECT_C = (255, 230, 120)
CONTROL_C = (150, 150, 160)
TEXT_C = (230, 230, 235)


def run_interactive(cfg: Config, seed: int = 0):
    import pygame  # imported here so headless modes never need pygame

    rng = np.random.default_rng(seed)
    world = World(cfg, rng)
    world.populate_random(cfg.start_population)

    # designate one creature as a frozen, untaught control for comparison
    control = world.creatures[0]
    control.brain.freeze()
    control._is_control = True

    selected = world.creatures[1] if len(world.creatures) > 1 else None

    pygame.init()
    screen = pygame.display.set_mode((int(cfg.world_width), int(cfg.world_height)))
    pygame.display.set_caption("A-Life creature sim -- click to select, hold R/P to teach")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    # rolling food-rate history for the selected creature and the control
    history_taught: list[float] = []
    history_control: list[float] = []
    metric_log: list[tuple] = []
    last_taught_food = 0
    last_control_food = 0
    METRIC_EVERY = 60  # ticks per metric sample

    running = True
    paused = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                c = world.creature_at(*event.pos)
                if c is not None:
                    selected = c
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if selected is not None and selected.alive:
            if keys[pygame.K_r]:
                selected.reward(cfg.reward_key_dose)
            if keys[pygame.K_p]:
                selected.punish(cfg.punish_key_dose)

        if not paused:
            world.step()
            # keep the demo populated so there is always something to teach
            if world.n_alive < 3:
                world.populate_random(cfg.start_population - world.n_alive)

            if world.tick % METRIC_EVERY == 0:
                if selected is not None:
                    rate = (selected.food_eaten - last_taught_food)
                    last_taught_food = selected.food_eaten
                    history_taught.append(rate)
                    history_taught[:] = history_taught[-200:]
                if control.alive:
                    crate = control.food_eaten - last_control_food
                    last_control_food = control.food_eaten
                    history_control.append(crate)
                    history_control[:] = history_control[-200:]
                metric_log.append((world.tick,
                                   selected.food_eaten if selected else 0,
                                   control.food_eaten if control.alive else last_control_food))

        # ---- draw ----
        screen.fill(BG)
        for fx, fy in world.food:
            pygame.draw.circle(screen, FOOD_C, (int(fx), int(fy)), int(cfg.food_radius))
        for hx, hy in world.hazards:
            pygame.draw.circle(screen, HAZARD_C, (int(hx), int(hy)),
                               int(cfg.hazard_radius), 2)
        for c in world.creatures:
            col = c.color
            pygame.draw.circle(screen, col, (int(c.x), int(c.y)), int(c.radius))
            hx = c.x + np.cos(c.heading) * c.radius
            hy = c.y + np.sin(c.heading) * c.radius
            pygame.draw.line(screen, (10, 10, 10), (c.x, c.y), (hx, hy), 2)
            if getattr(c, "_is_control", False):
                pygame.draw.circle(screen, CONTROL_C, (int(c.x), int(c.y)),
                                   int(c.radius) + 3, 1)
            if c is selected:
                pygame.draw.circle(screen, SELECT_C, (int(c.x), int(c.y)),
                                   int(c.radius) + 4, 2)

        _draw_hud(screen, font, world, selected, control,
                  history_taught, history_control, cfg)
        pygame.display.flip()
        clock.tick(cfg.tick_rate)

    pygame.quit()

    # dump the teaching metric so the learning curve is plottable
    if metric_log:
        with open("teaching_metric.csv", "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(["tick", "selected_food_total", "control_food_total"])
            w.writerows(metric_log)
        print("wrote teaching_metric.csv")


def _draw_hud(screen, font, world, selected, control, hist_t, hist_c, cfg):
    import pygame
    lines = [
        "click: select   hold R: reward   hold P: punish   SPACE: pause   ESC: quit",
        f"tick {world.tick}   alive {world.n_alive}",
    ]
    if selected is not None:
        b = selected.biochem
        lines.append(f"selected #{selected.id}: food {selected.food_eaten}  "
                     f"glucose {b.glucose:5.1f}  hunger {b.hunger:.2f}  "
                     f"reward {b.reward:.2f}  punish {b.punishment:.2f}  "
                     f"lr {selected.brain.lr:.3f}")
    lines.append(f"control #{control.id}: food {control.food_eaten} (plasticity OFF)")

    y = 6
    for ln in lines:
        screen.blit(font.render(ln, True, TEXT_C), (8, y))
        y += 16

    # tiny sparkline of taught (yellow) vs control (grey) food rate
    _sparkline(screen, hist_t, SELECT_C, base_y=cfg.world_height - 12, scale=10)
    _sparkline(screen, hist_c, CONTROL_C, base_y=cfg.world_height - 12, scale=10)


def _sparkline(screen, hist, color, base_y, scale):
    import pygame
    if len(hist) < 2:
        return
    x0 = 8
    pts = [(x0 + i * 3, base_y - min(80, v * scale)) for i, v in enumerate(hist[-200:])]
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 1)

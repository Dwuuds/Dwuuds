"""Mode 1: the interactive teaching viewer (proves within-life learning).

A polished real-time window built on pygame. The left two-thirds is the living
world; the right panel is a live inspector for the selected creature -- its
biochemistry, its drives, and a real-time picture of its neural net firing and
re-wiring as you teach it. A chart at the bottom of the panel tracks the taught
creature's food-rate against a frozen "control" so you can watch it pull ahead.

Rendering is deliberately simple in spirit (circles, lines, bars) but composed to
actually look like something. The simulation core (world/creature/brain/biochem)
is untouched and fully decoupled -- this file only draws it and routes input.

Controls
  click / TAB     select a creature (TAB cycles)
  hold R          reward the selected creature   (teach: "yes, like that")
  hold P          punish the selected creature   (teach: "no, not that")
  SPACE           pause / resume
  [ ]             slow down / speed up the simulation
  F               toggle follow-cam highlight on the selected creature
  N               spawn a fresh random creature
  C               jump selection to the control creature
  S               save a screenshot
  ESC             quit (writes teaching_metric.csv)
"""

from __future__ import annotations

import csv as csvmod
import math
import os
from collections import deque, defaultdict

import numpy as np
import pygame

from config import Config
from world import World
from brain import SENSOR_NAMES, MOTOR_NAMES

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
BG          = (16, 18, 27)
WORLD_BG    = (22, 25, 38)
GRID        = (30, 34, 50)
PANEL_BG    = (26, 29, 44)
PANEL_EDGE  = (44, 49, 72)
HEADER_BG   = (12, 14, 22)
TEXT        = (222, 226, 240)
TEXT_DIM    = (132, 140, 168)
ACCENT      = (74, 200, 217)
SELECT_C    = (255, 210, 74)
CONTROL_C   = (150, 156, 178)
FOOD_C      = (120, 220, 130)
FOOD_GLOW   = (60, 140, 80)
HAZARD_C    = (228, 84, 84)
HAZARD_GLOW = (120, 40, 44)
REWARD_C    = (96, 220, 130)
PUNISH_C    = (236, 96, 104)
POS_W       = (90, 200, 220)
NEG_W       = (220, 110, 180)

# layout
HEADER_H = 46
MARGIN   = 16
PANEL_W  = 432


def _aa_filled_circle(surf, color, cx, cy, r):
    """Anti-aliased filled circle with a graceful fallback."""
    cx, cy, r = int(cx), int(cy), int(r)
    if r <= 0:
        return
    try:
        import pygame.gfxdraw as gfx
        gfx.filled_circle(surf, cx, cy, r, color)
        gfx.aacircle(surf, cx, cy, r, color)
    except Exception:
        pygame.draw.circle(surf, color, (cx, cy), r)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _activation_color(v):
    """Map a neuron activation in [-1,1] to a color: blue (neg) -> dim -> orange."""
    v = max(-1.0, min(1.0, float(v)))
    if v >= 0:
        return _lerp((54, 58, 82), (255, 176, 64), v)
    return _lerp((54, 58, 82), (74, 150, 240), -v)


class Renderer:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.world = World(cfg, np.random.default_rng(seed))
        self.world.populate_random(cfg.start_population)

        # one creature is a frozen, untaught control for comparison
        self.control = self.world.creatures[0]
        self.control.brain.freeze()
        self.control._is_control = True
        self.selected = self.world.creatures[1] if len(self.world.creatures) > 1 else self.control

        self.win_w = int(cfg.world_width) + 3 * MARGIN + PANEL_W
        self.win_h = HEADER_H + 2 * MARGIN + int(cfg.world_height)
        self.world_x = MARGIN
        self.world_y = HEADER_H + MARGIN
        self.panel_x = self.world_x + int(cfg.world_width) + MARGIN
        self.panel_y = self.world_y
        self.panel_h = int(cfg.world_height)

        self.paused = False
        self.follow = False
        self.steps_per_frame = 1
        self.frame = 0

        # teaching feedback (decaying glow when R/P pressed)
        self.reward_flash = 0.0
        self.punish_flash = 0.0

        # trails per creature id
        self.trails = defaultdict(lambda: deque(maxlen=22))

        # performance history (cumulative food since the current creature was
        # selected, vs the control over the same span -- clean diverging lines)
        self.METRIC_EVERY = 30
        self.hist_taught = deque(maxlen=240)
        self.hist_control = deque(maxlen=240)
        self._sel_base_taught = self.selected.food_eaten
        self._sel_base_control = self.control.food_eaten
        self._last_selected_id = self.selected.id
        self.metric_log = []

        self.screens_dir = "screenshots"

    # ---- fonts (lazy, after display init) -----------------------------
    def _init_fonts(self):
        pygame.font.init()
        self.f_title = pygame.font.SysFont("dejavusans,arial", 20, bold=True)
        self.f_h = pygame.font.SysFont("dejavusans,arial", 15, bold=True)
        self.f = pygame.font.SysFont("dejavusansmono,consolas,monospace", 13)
        self.f_sm = pygame.font.SysFont("dejavusansmono,consolas,monospace", 11)

    # ---- selection helpers -------------------------------------------
    def _cycle_selection(self, step=1):
        alive = [c for c in self.world.creatures if c.alive]
        if not alive:
            return
        if self.selected in alive:
            i = alive.index(self.selected)
            self.selected = alive[(i + step) % len(alive)]
        else:
            self.selected = alive[0]

    # ---- input --------------------------------------------------------
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            wx = event.pos[0] - self.world_x
            wy = event.pos[1] - self.world_y
            c = self.world.creature_at(wx, wy)
            if c is not None:
                self.selected = c
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE,):
                return False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_TAB:
                self._cycle_selection(1)
            elif event.key == pygame.K_f:
                self.follow = not self.follow
            elif event.key == pygame.K_c:
                self.selected = self.control
            elif event.key == pygame.K_n:
                self.world.populate_random(1)
            elif event.key == pygame.K_LEFTBRACKET:
                self.steps_per_frame = max(1, self.steps_per_frame - 1)
            elif event.key == pygame.K_RIGHTBRACKET:
                self.steps_per_frame = min(8, self.steps_per_frame + 1)
            elif event.key == pygame.K_s:
                self.save_screenshot()
        return True

    def _apply_teaching(self):
        keys = pygame.key.get_pressed()
        if self.selected is not None and self.selected.alive:
            if keys[pygame.K_r]:
                self.selected.reward(self.cfg.reward_key_dose)
                self.reward_flash = 1.0
            if keys[pygame.K_p]:
                self.selected.punish(self.cfg.punish_key_dose)
                self.punish_flash = 1.0

    # ---- update -------------------------------------------------------
    def update(self):
        if self.paused:
            return
        target = self.cfg.start_population
        for _ in range(self.steps_per_frame):
            # keep the control alive as a persistent (mildly hungry) baseline, and
            # keep your current pupil alive so you can teach it without it starving
            # out from under you -- both stay hungry enough to stay motivated
            self._sustain(self.control)
            self._sustain(self.selected)
            self.world.step()
            # keep the sandbox lively
            if self.world.n_alive < int(target * 0.6):
                self.world.populate_random(target - self.world.n_alive)
            if self.world.tick % self.METRIC_EVERY == 0:
                self._sample_metrics()
        # if the creature you were watching died, follow a new learner
        if self.selected is None or not self.selected.alive:
            self._cycle_selection(1)
        self._check_selection_changed()
        # trails
        for c in self.world.creatures:
            if c.alive:
                self.trails[c.id].append((c.x, c.y))
        self.reward_flash *= 0.90
        self.punish_flash *= 0.90

    def _sustain(self, c):
        """Keep a creature from dying without changing how it forages: top up only
        enough glucose to avoid starvation (it stays hungry and motivated) and
        hold the aging clock just under the cap."""
        if c is None:
            return
        c.alive = True
        if c.biochem.glucose < 25.0:
            c.biochem.glucose = 25.0
        c.biochem.age = min(c.biochem.age, self.cfg.max_age_ticks - 1)

    def _check_selection_changed(self):
        if self.selected is not None and self.selected.id != self._last_selected_id:
            self._last_selected_id = self.selected.id
            self._sel_base_taught = self.selected.food_eaten
            self._sel_base_control = self.control.food_eaten
            self.hist_taught.clear()
            self.hist_control.clear()

    def _sample_metrics(self):
        if self.selected is not None:
            self.hist_taught.append(self.selected.food_eaten - self._sel_base_taught)
        self.hist_control.append(self.control.food_eaten - self._sel_base_control)
        self.metric_log.append((self.world.tick,
                                self.selected.food_eaten if self.selected else 0,
                                self.control.food_eaten))

    # ---- drawing: world ----------------------------------------------
    def _draw_world(self, screen):
        view = pygame.Surface((int(self.cfg.world_width), int(self.cfg.world_height)))
        view.fill(WORLD_BG)
        # subtle grid
        for gx in range(0, int(self.cfg.world_width), 40):
            pygame.draw.line(view, GRID, (gx, 0), (gx, self.cfg.world_height))
        for gy in range(0, int(self.cfg.world_height), 40):
            pygame.draw.line(view, GRID, (0, gy), (self.cfg.world_width, gy))

        # hazards (pulsing)
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.06)
        for hx, hy in self.world.hazards:
            _aa_filled_circle(view, HAZARD_GLOW, hx, hy, self.cfg.hazard_radius + 6 * pulse)
            _aa_filled_circle(view, _lerp(HAZARD_GLOW, HAZARD_C, 0.6), hx, hy, self.cfg.hazard_radius)
            pygame.draw.circle(view, HAZARD_C, (int(hx), int(hy)),
                               int(self.cfg.hazard_radius), 2)

        # food (soft glow)
        for fx, fy in self.world.food:
            _aa_filled_circle(view, FOOD_GLOW, fx, fy, self.cfg.food_radius + 3)
            _aa_filled_circle(view, FOOD_C, fx, fy, self.cfg.food_radius)

        # trails
        for c in self.world.creatures:
            tr = self.trails.get(c.id)
            if tr and len(tr) > 2:
                col = c.color
                n = len(tr)
                for i in range(1, n):
                    a = i / n
                    pygame.draw.line(view, _lerp(WORLD_BG, col, 0.25 * a),
                                     tr[i - 1], tr[i], 1)

        # creatures
        for c in self.world.creatures:
            self._draw_creature(view, c)

        screen.blit(view, (self.world_x, self.world_y))
        pygame.draw.rect(screen, PANEL_EDGE,
                         (self.world_x - 1, self.world_y - 1,
                          self.cfg.world_width + 2, self.cfg.world_height + 2), 1)

    def _draw_creature(self, view, c):
        x, y, r = c.x, c.y, c.radius
        is_sel = c is self.selected
        is_ctrl = getattr(c, "_is_control", False)

        # teaching glow on the selected creature
        if is_sel and (self.reward_flash > 0.05 or self.punish_flash > 0.05):
            if self.reward_flash >= self.punish_flash:
                glow, mag = REWARD_C, self.reward_flash
            else:
                glow, mag = PUNISH_C, self.punish_flash
            _aa_filled_circle(view, _lerp(WORLD_BG, glow, 0.5 * mag), x, y, r + 10 * mag + 4)

        # energy ring (glucose fraction)
        frac = max(0.0, min(1.0, c.biochem.glucose / self.cfg.glucose_max))
        if frac > 0:
            ring_col = _lerp(HAZARD_C, FOOD_C, frac)
            self._draw_arc(view, ring_col, x, y, r + 4, frac)

        # body
        body = c.color
        _aa_filled_circle(view, _lerp((0, 0, 0), body, 0.55), x, y, r + 1)
        _aa_filled_circle(view, body, x, y, r)

        # heading + eyes
        ch, sh = math.cos(c.heading), math.sin(c.heading)
        # eyes
        ex, ey = x + ch * r * 0.45, y + sh * r * 0.45
        perp = (-sh, ch)
        for s in (-1, 1):
            eye_x = ex + perp[0] * r * 0.4 * s
            eye_y = ey + perp[1] * r * 0.4 * s
            _aa_filled_circle(view, (245, 245, 250), eye_x, eye_y, max(1.5, r * 0.22))
            _aa_filled_circle(view, (20, 20, 28), eye_x + ch * 1.2, eye_y + sh * 1.2,
                              max(1.0, r * 0.11))
        # eat indicator
        try:
            eat = float(c.brain._out[MOTOR_NAMES.index("eat")])
        except Exception:
            eat = 0.0
        if eat > 0.2:
            _aa_filled_circle(view, (30, 12, 14), x + ch * r * 0.9, y + sh * r * 0.9,
                              max(1.5, r * 0.25))

        # selection / control rings
        if is_ctrl:
            self._dashed_circle(view, CONTROL_C, x, y, r + 6)
        if is_sel:
            pygame.draw.circle(view, SELECT_C, (int(x), int(y)), int(r + 8), 2)
            pygame.draw.line(view, SELECT_C, (x, y - r - 12), (x, y - r - 6), 2)

    def _draw_arc(self, surf, color, cx, cy, r, frac):
        pts = []
        steps = max(6, int(36 * frac))
        for i in range(steps + 1):
            ang = -math.pi / 2 + (2 * math.pi * frac) * (i / steps)
            pts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
        if len(pts) >= 2:
            pygame.draw.lines(surf, color, False, pts, 2)

    def _dashed_circle(self, surf, color, cx, cy, r, dashes=16):
        for i in range(dashes):
            if i % 2 == 0:
                a0 = 2 * math.pi * i / dashes
                a1 = 2 * math.pi * (i + 0.6) / dashes
                pygame.draw.line(surf, color,
                                 (cx + math.cos(a0) * r, cy + math.sin(a0) * r),
                                 (cx + math.cos(a1) * r, cy + math.sin(a1) * r), 1)

    # ---- drawing: header ---------------------------------------------
    def _draw_header(self, screen):
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.win_w, HEADER_H))
        pygame.draw.line(screen, PANEL_EDGE, (0, HEADER_H), (self.win_w, HEADER_H))
        screen.blit(self.f_title.render("A-LIFE  ·  creature sim", True, ACCENT), (16, 12))
        speed = "PAUSED" if self.paused else f"{self.steps_per_frame}x"
        info = (f"alive {self.world.n_alive:>2}   "
                f"tick {self.world.tick:>6}   food {len(self.world.food):>3}   "
                f"sim {speed:>6}")
        surf = self.f.render(info, True, TEXT_DIM)
        screen.blit(surf, (self.win_w - surf.get_width() - 16, 16))

    # ---- drawing: panel ----------------------------------------------
    def _draw_panel(self, screen):
        px, py, pw, ph = self.panel_x, self.panel_y, PANEL_W, self.panel_h
        pygame.draw.rect(screen, PANEL_BG, (px, py, pw, ph), border_radius=8)
        pygame.draw.rect(screen, PANEL_EDGE, (px, py, pw, ph), 1, border_radius=8)

        c = self.selected
        x = px + 16
        y = py + 12
        if c is None:
            screen.blit(self.f, (x, y))
            return

        is_ctrl = getattr(c, "_is_control", False)
        badge = "CONTROL · learning OFF" if is_ctrl else "LEARNER · plastic"
        badge_c = CONTROL_C if is_ctrl else REWARD_C
        # title row with color swatch
        pygame.draw.rect(screen, c.color, (x, y + 2, 16, 16), border_radius=4)
        screen.blit(self.f_h.render(f"creature #{c.id}", True, TEXT), (x + 24, y))
        bs = self.f_sm.render(badge, True, badge_c)
        screen.blit(bs, (px + pw - bs.get_width() - 16, y + 2))
        y += 26
        screen.blit(self.f_sm.render(
            f"gen {c.generation}   age {c.age_ticks}   food eaten {c.food_eaten}",
            True, TEXT_DIM), (x, y))
        y += 22

        # vital bars
        b = c.biochem
        bar_w = pw - 32
        y = self._bar(screen, x, y, bar_w, "energy", b.glucose / self.cfg.glucose_max,
                      _lerp(HAZARD_C, FOOD_C, min(1, b.glucose / self.cfg.glucose_max)))
        y = self._bar(screen, x, y, bar_w, "hunger", min(1, b.hunger), (230, 170, 70))
        y = self._bar(screen, x, y, bar_w, "reward", min(1, b.reward / self.cfg.chem_max), REWARD_C)
        y = self._bar(screen, x, y, bar_w, "punish", min(1, b.punishment / self.cfg.chem_max), PUNISH_C)
        y = self._bar(screen, x, y, bar_w, "pain", min(1, b.pain / self.cfg.chem_max), (210, 90, 150))
        y = self._bar(screen, x, y, bar_w, "fatigue", min(1, b.fatigue), (150, 140, 210))
        y += 6

        # learning-rule readout
        mod = b.modulator
        modc = REWARD_C if mod >= 0 else PUNISH_C
        screen.blit(self.f_sm.render("three-factor learning", True, TEXT_DIM), (x, y))
        ms = self.f_sm.render(f"lr {c.brain.lr:.3f}   modulator {mod:+.2f}", True, modc)
        screen.blit(ms, (px + pw - ms.get_width() - 16, y))
        y += 18

        # brain visualization
        brain_h = 168
        y = self._draw_brain(screen, x, y, bar_w, brain_h, c)
        y += 8

        # performance chart
        self._draw_chart(screen, x, y, bar_w, py + ph - y - 56)

        # controls footer
        self._draw_controls(screen, x, py + ph - 44, bar_w)

    def _bar(self, screen, x, y, w, label, frac, color):
        frac = max(0.0, min(1.0, frac))
        screen.blit(self.f_sm.render(label, True, TEXT_DIM), (x, y))
        bx = x + 64
        bw = w - 64
        pygame.draw.rect(screen, (40, 44, 64), (bx, y + 1, bw, 11), border_radius=5)
        if frac > 0:
            pygame.draw.rect(screen, color, (bx, y + 1, max(2, int(bw * frac)), 11),
                             border_radius=5)
        return y + 17

    def _draw_brain(self, screen, x, y, w, h, c):
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, (20, 22, 34), box, border_radius=6)
        pygame.draw.rect(screen, PANEL_EDGE, box, 1, border_radius=6)
        screen.blit(self.f_sm.render("brain  (live activations + synapse weights)",
                                     True, TEXT_DIM), (x + 8, y + 4))

        brain = c.brain
        xin = np.asarray(brain._x)
        hid = np.asarray(brain._hidden)
        out = np.asarray(brain._out)
        nin, nh, no = len(xin), len(hid), len(out)

        pad_top = y + 22
        pad_bot = y + h - 10
        col_in = x + 26
        col_hid = x + w // 2
        col_out = x + w - 26

        def ys(n):
            if n == 1:
                return [(pad_top + pad_bot) / 2]
            return [pad_top + (pad_bot - pad_top) * i / (n - 1) for i in range(n)]

        yin, yhid, yout = ys(nin), ys(nh), ys(no)

        # weight lines on an alpha surface
        wsurf = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        bound = max(1e-6, c.brain.w_max)
        w1 = brain.w1
        for j in range(nh):
            for i in range(nin):
                wv = w1[j, i] / bound
                a = min(150, int(abs(wv) * 150))
                if a < 18:
                    continue
                col = (POS_W if wv >= 0 else NEG_W) + (a,)
                pygame.draw.line(wsurf, col, (col_in, yin[i]), (col_hid, yhid[j]), 1)
        w2 = brain.w2
        for k in range(no):
            for j in range(nh):
                wv = w2[k, j] / bound
                a = min(170, int(abs(wv) * 170))
                if a < 18:
                    continue
                col = (POS_W if wv >= 0 else NEG_W) + (a,)
                pygame.draw.line(wsurf, col, (col_hid, yhid[j]), (col_out, yout[k]), 1)
        screen.blit(wsurf, (0, 0))

        # nodes
        for i in range(nin):
            _aa_filled_circle(screen, _activation_color(xin[i]), col_in, yin[i], 4)
        for j in range(nh):
            _aa_filled_circle(screen, _activation_color(hid[j]), col_hid, yhid[j], 4)
        for k in range(no):
            _aa_filled_circle(screen, _activation_color(out[k]), col_out, yout[k], 6)

        # motor labels
        for k, name in enumerate(MOTOR_NAMES):
            lab = {"wheel_left": "L", "wheel_right": "R", "eat": "eat"}[name]
            screen.blit(self.f_sm.render(lab, True, TEXT_DIM),
                        (col_out + 8, yout[k] - 6))
        screen.blit(self.f_sm.render("sensors", True, TEXT_DIM), (x + 8, pad_bot + -2))
        return y + h

    def _draw_chart(self, screen, x, y, w, h):
        h = max(48, h)
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, (20, 22, 34), box, border_radius=6)
        pygame.draw.rect(screen, PANEL_EDGE, box, 1, border_radius=6)
        t_tot = self.hist_taught[-1] if self.hist_taught else 0
        c_tot = self.hist_control[-1] if self.hist_control else 0
        screen.blit(self.f_sm.render("food eaten since selected", True, TEXT_DIM), (x + 8, y + 4))
        lbl = self.f_sm.render(f"taught {t_tot}", True, SELECT_C)
        screen.blit(lbl, (x + w - lbl.get_width() - 8, y + 4))
        lbl2 = self.f_sm.render(f"control {c_tot}", True, CONTROL_C)
        screen.blit(lbl2, (x + w - lbl2.get_width() - lbl.get_width() - 16, y + 4))

        # shared vertical scale so the divergence between the two lines is honest
        mx = max(1.0, float(max(list(self.hist_taught) + list(self.hist_control) + [1])))
        base = y + h - 8
        top = y + 22

        def plot(hist, color):
            if len(hist) < 2:
                return
            data = list(hist)
            n = len(data)
            pts = [(x + 8 + (w - 16) * (i / (n - 1)), base - (base - top) * (v / mx))
                   for i, v in enumerate(data)]
            pygame.draw.lines(screen, color, False, pts, 2)

        plot(self.hist_control, CONTROL_C)
        plot(self.hist_taught, SELECT_C)

    def _draw_controls(self, screen, x, y, w):
        line1 = "click/TAB select   hold R reward   hold P punish"
        line2 = "SPACE pause   [ ] speed   N spawn   C control   S shot   ESC quit"
        screen.blit(self.f_sm.render(line1, True, TEXT_DIM), (x, y))
        screen.blit(self.f_sm.render(line2, True, TEXT_DIM), (x, y + 16))

    # ---- compose ------------------------------------------------------
    def draw(self, screen):
        screen.fill(BG)
        self._draw_header(screen)
        self._draw_world(screen)
        self._draw_panel(screen)
        self.frame += 1

    def save_screenshot(self, path=None):
        os.makedirs(self.screens_dir, exist_ok=True)
        if path is None:
            path = os.path.join(self.screens_dir, f"shot_{self.frame:05d}.png")
        pygame.image.save(self._last_screen, path)
        print(f"saved {path}")
        return path

    # ---- main loop ----------------------------------------------------
    def run(self):
        pygame.init()
        self._init_fonts()
        screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("A-Life creature sim — teach a creature to feed itself")
        clock = pygame.time.Clock()
        self._last_screen = screen
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self._apply_teaching()
            self.update()
            self.draw(screen)
            pygame.display.flip()
            clock.tick(self.cfg.tick_rate)
        self._dump_metrics()
        pygame.quit()

    def _dump_metrics(self):
        if not self.metric_log:
            return
        with open("teaching_metric.csv", "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(["tick", "selected_food_total", "control_food_total"])
            w.writerows(self.metric_log)
        print("wrote teaching_metric.csv")


def run_interactive(cfg: Config, seed: int = 0):
    Renderer(cfg, seed).run()

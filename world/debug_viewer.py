"""
Live debug viewer for the RL environment.

Active when gui=True. Creates a pygame window and renders the environment
at each training step. Use it as a drop-in wrapper around Environment:

    env = DebugViewer(env)   # wraps env; no-op when gui=False

Controls:
  Space         — pause / resume
  T             — toggle "pause on target reached"
  C             — toggle "pause on collision"
  ← →           — scrub through step history while paused
                  (→ when not scrubbing: advance exactly one live step)
  Drag slider   — change simulation speed (1–60 fps)
  Click HUD     — same as T / C toggles
"""
import collections
import math
import numpy as np
import pygame

from world.helpers import ACTIONS
from world.state import SENSOR_DIRECTIONS, ObservationBuilder

_ENV_SIZE = 650   # square pixels for the environment viewport
_HUD_W    = 220   # right-side panel width

_BG        = (245, 245, 245)
_OBSTACLE  = (80,  80,  80)
_TARGET    = (40,  200, 80)
_AGENT     = (30,  100, 220)
_PATH      = (0,   180, 240)
_SCRUB_DOT = (255, 140,   0)
_HUD_BG    = (20,  20,  20)
_TEXT      = (220, 220, 220)
_DIM       = (130, 130, 130)
_ON        = (50,  170,  60)
_OFF       = (140,  40,  40)
_PAUSE_CLR = (220, 160,   0)
_RUN_CLR   = (60,  210,  60)


class DebugViewer:
    """Wraps Environment and adds a live pygame debug view when no_gui=False."""

    def __init__(self, env, history_len=2000):
        self.env    = env
        self.active = not env.no_gui

        # Metrics pushed from the training loop via update_metrics()
        self.epsilon = None
        self.loss    = None

        self._episode = 0
        self._step    = 0
        self._cum_r   = 0.0
        self._path    = []
        self._history = collections.deque(maxlen=history_len)
        self._scrub   = None    # None = live mode; int = index into _history
        self._paused  = False

        self.pause_on_target    = False
        self.pause_on_collision = False

        self._last_action    = None   # action index from last step
        self._flash_frames   = 0      # frames left for collision flash
        self.max_q           = None   # pushed from training loop
        self._targets_reached= 0      # cumulative target reaches across all episodes

        self._speed         = 15     # rendering FPS (slider-controlled)
        self._dragging_speed= False
        self._slider_rect   = None

        self._btn_target    = None
        self._btn_collision = None

        if self.active:
            pygame.init()
            self._screen = pygame.display.set_mode((_ENV_SIZE + _HUD_W, _ENV_SIZE))
            pygame.display.set_caption("RL Debug Viewer")
            self._font   = pygame.font.SysFont("monospace", 13)
            self._font_s = pygame.font.SysFont("monospace", 11)
            self._clock  = pygame.time.Clock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._path    = [self.env.agent_pos]
        self._cum_r   = 0.0
        self._step    = 0
        self._scrub   = None
        self._episode += 1
        if self.active:
            self._render()
        return result

    def step(self, action):
        self._last_action = action
        result = self.env.step(action)
        _, reward, done, info = result

        self._cum_r += reward
        self._step  += 1
        self._path.append(self.env.agent_pos)
        if info.get('collided'):
            self._flash_frames = 8
        if info.get('target_reached'):
            self._targets_reached += 1

        self._history.append({
            'pos':     self.env.agent_pos,
            'episode': self._episode,
            'step':    self._step,
            'cum_r':   self._cum_r,
            'reward':  reward,
            'done':    done,
            'info':    info,
            'drifted': 'random move' in info.get('actual_action', ''),
            'action':  action,
        })

        if self.active:
            self._render()
            self._process_events()
            if self.pause_on_target    and info.get('target_reached'): self._paused = True
            if self.pause_on_collision and info.get('collided'):        self._paused = True
            if self._paused:
                self._wait_for_input()

        return result

    def update_metrics(self, epsilon=None, loss=None, max_q=None):
        """Call after each training step to update the HUD values."""
        if epsilon is not None: self.epsilon = epsilon
        if loss    is not None: self.loss    = loss
        if max_q   is not None: self.max_q   = max_q

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _to_screen(self, x, y):
        px = int(x / self.env.x_max * _ENV_SIZE)
        py = int(_ENV_SIZE - y / self.env.y_max * _ENV_SIZE)
        return px, py

    def _render(self, scrub=False):
        env  = self.env
        surf = self._screen
        surf.fill(_BG)

        # obstacles
        for ox, oy, ow, oh in env.obstacles:
            sx, sy = self._to_screen(ox, oy + oh)
            sw = max(1, int(ow / env.x_max * _ENV_SIZE))
            sh = max(1, int(oh / env.y_max * _ENV_SIZE))
            pygame.draw.rect(surf, _OBSTACLE, (sx, sy, sw, sh))

        # target
        if env.target_pos:
            tr = max(4, int(env.target_radius / env.x_max * _ENV_SIZE))
            pygame.draw.circle(surf, _TARGET, self._to_screen(*env.target_pos), tr)

        # path trace
        path = self._scrub_path() if (scrub and self._scrub is not None) else self._path
        if len(path) >= 2:
            pygame.draw.lines(surf, _PATH, False, [self._to_screen(*p) for p in path], 2)

        # drift markers — orange × at positions where stochastic kick occurred
        scrub_entry = self._history[self._scrub] if (scrub and self._scrub is not None) else None
        cur_ep  = scrub_entry['episode'] if scrub_entry else self._episode
        cur_stp = scrub_entry['step']    if scrub_entry else self._step
        for entry in self._history:
            if entry['episode'] == cur_ep and entry['step'] <= cur_stp and entry.get('drifted'):
                mx, my = self._to_screen(*entry['pos'])
                pygame.draw.line(surf, (255, 130, 0), (mx - 5, my - 5), (mx + 5, my + 5), 2)
                pygame.draw.line(surf, (255, 130, 0), (mx - 5, my + 5), (mx + 5, my - 5), 2)

        # sensor rays — shown in both live and scrub mode
        apos = self._history[self._scrub]['pos'] if (scrub and self._scrub is not None) else env.agent_pos
        if apos:
            self._draw_sensor_rays(surf, apos)

        # action arrow — use historical action when scrubbing
        action_to_draw = scrub_entry.get('action') if scrub_entry else self._last_action
        if apos and action_to_draw is not None:
            self._draw_action_arrow(surf, apos, action_to_draw)

        # agent — flash red on collision, orange when scrubbing
        if apos:
            ar = max(3, int(env.agent_radius / env.x_max * _ENV_SIZE))
            if scrub and self._scrub is not None:
                col = _SCRUB_DOT
            elif self._flash_frames > 0:
                col = (220, 50, 50)
                self._flash_frames -= 1
            else:
                col = _AGENT
            pygame.draw.circle(surf, col, self._to_screen(*apos), ar)

        # border
        pygame.draw.rect(surf, (0, 0, 0), (0, 0, _ENV_SIZE, _ENV_SIZE), 2)

        self._draw_hud(scrub)
        pygame.display.flip()
        self._clock.tick(self._speed)

    def _draw_hud(self, scrub=False):
        panel = pygame.Surface((_HUD_W, _ENV_SIZE))
        panel.fill(_HUD_BG)

        y = 10
        def line(text, color=_TEXT, f=None):
            nonlocal y
            img = (f or self._font).render(text, True, color)
            panel.blit(img, (8, y))
            y += img.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(panel, (60, 60, 60), (8, y), (_HUD_W - 8, y))
            y += 8

        if scrub and self._scrub is not None:
            h = self._history[self._scrub]
            line("SCRUBBING", _PAUSE_CLR)
            line(f"Ep    {h['episode']}")
            line(f"Step  {h['step']}")
            line(f"Rew   {h['reward']:+.2f}")
            line(f"CumR  {h['cum_r']:.1f}")
            if h['pos']:
                line(f"Pos   ({h['pos'][0]:.2f},{h['pos'][1]:.2f})")
        else:
            line("  PAUSED  " if self._paused else "  RUNNING ", _PAUSE_CLR if self._paused else _RUN_CLR)
            line(f"Ep    {self._episode}")
            line(f"Step  {self._step}")
            line(f"CumR  {self._cum_r:.1f}")
            pos = self.env.agent_pos
            if pos:
                line(f"Pos   ({pos[0]:.2f},{pos[1]:.2f})")
            if self.epsilon is not None: line(f"Eps   {self.epsilon:.4f}")
            if self.loss    is not None: line(f"Loss  {self.loss:.6f}")
            if self.max_q   is not None: line(f"MaxQ  {self.max_q:.2f}")
            rate = 100.0 * self._targets_reached / max(1, self._episode)
            line(f"Goals {self._targets_reached} ({rate:.0f}%)",
                 (80, 220, 100) if rate >= 50 else _TEXT)
            drift_n = sum(1 for e in self._history
                          if e['episode'] == self._episode and e.get('drifted'))
            if drift_n: line(f"Drift {drift_n}", (255, 150, 50))

        sep()
        line("Auto-pause", _DIM, self._font_s)

        for label, attr, key in [
            ("On target  [T]",    "pause_on_target",    "_btn_target"),
            ("On collision  [C]", "pause_on_collision", "_btn_collision"),
        ]:
            btn = pygame.Rect(8, y, _HUD_W - 16, 22)
            setattr(self, key, btn)
            pygame.draw.rect(panel, _ON if getattr(self, attr) else _OFF, btn, border_radius=3)
            panel.blit(self._font_s.render(label, True, _TEXT), (12, y + 4))
            y += 26

        sep()
        line("Space  pause/resume", _DIM, self._font_s)
        line("← →    scrub history", _DIM, self._font_s)

        sep()
        # Speed slider row: "Speed" label | track+handle | "N fps"
        slider_y = y + 2
        panel.blit(self._font_s.render("Speed", True, _DIM), (8, slider_y))
        track_x = 52
        track_w = _HUD_W - track_x - 38
        pygame.draw.rect(panel, (60, 60, 60), (track_x, slider_y + 2, track_w, 6), border_radius=3)
        ratio = (self._speed - 1) / 59.0
        hx = track_x + int(ratio * track_w)
        pygame.draw.circle(panel, (180, 180, 255), (hx, slider_y + 5), 6)
        panel.blit(self._font_s.render(f"{self._speed:2d}fps", True, _TEXT),
                   (track_x + track_w + 4, slider_y))
        # Store hit rect in screen coords for mouse events
        self._slider_rect = pygame.Rect(_ENV_SIZE + track_x, slider_y + 2, track_w, 14)

        self._screen.blit(panel, (_ENV_SIZE, 0))

    def _draw_sensor_rays(self, surf, apos):
        """Draw 8 LiDAR rays from agent position, green=far → red=close."""
        env = self.env
        obs = ObservationBuilder(env, "sensors", sensor_range=10.0)
        readings = obs.get_sensor_readings(*apos)
        ax, ay = self._to_screen(*apos)
        for (dx, dy), dist in zip(SENSOR_DIRECTIONS, readings):
            ratio = dist / 10.0  # 0=close, 1=far
            r = int(220 * (1 - ratio) + 50 * ratio)
            g = int(50  * (1 - ratio) + 200 * ratio)
            color = (r, g, 60)
            ex = apos[0] + dx * dist
            ey = apos[1] + dy * dist
            pygame.draw.line(surf, color, (ax, ay), self._to_screen(ex, ey), 1)

    def _draw_action_arrow(self, surf, apos, action_id):
        """Draw a direction arrow showing the last chosen action."""
        direction, step_size = ACTIONS[action_id]
        dx, dy = direction
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        # Scale for visibility: use step_size but minimum 0.8 world units
        vis_len = max(step_size, 0.8)
        ex = apos[0] + ux * vis_len
        ey = apos[1] + uy * vis_len
        start = self._to_screen(*apos)
        end   = self._to_screen(ex, ey)
        pygame.draw.line(surf, (255, 220, 0), start, end, 2)
        # Arrowhead
        angle = math.atan2(-(end[1] - start[1]), end[0] - start[0])
        for side in (math.pi * 5/6, -math.pi * 5/6):
            hx = end[0] + 8 * math.cos(angle + side)
            hy = end[1] - 8 * math.sin(angle + side)
            pygame.draw.line(surf, (255, 220, 0), end, (int(hx), int(hy)), 2)

    def _scrub_path(self):
        if self._scrub is None or not self._history:
            return self._path
        ep  = self._history[self._scrub]['episode']
        stp = self._history[self._scrub]['step']
        return [e['pos'] for e in self._history if e['episode'] == ep and e['step'] <= stp]

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: self._paused = not self._paused
                if event.key == pygame.K_t:     self.pause_on_target    = not self.pause_on_target
                if event.key == pygame.K_c:     self.pause_on_collision = not self.pause_on_collision
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)
            if event.type == pygame.MOUSEMOTION and self._dragging_speed:
                self._update_speed_from_mouse(event.pos[0])
            if event.type == pygame.MOUSEBUTTONUP:
                self._dragging_speed = False

    def _wait_for_input(self):
        while self._paused or self._scrub is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._paused = False
                        self._scrub  = None
                        self._render()
                        return

                    if event.key == pygame.K_RIGHT:
                        if self._scrub is not None and self._scrub < len(self._history) - 1:
                            self._scrub += 1
                        else:
                            # Exit scrub and do exactly one live step, then re-pause.
                            # _paused stays True so step() calls _wait_for_input() again.
                            self._scrub = None
                            return

                    if event.key == pygame.K_LEFT:
                        if self._scrub is None:
                            self._scrub = max(0, len(self._history) - 2)
                        elif self._scrub > 0:
                            self._scrub -= 1

                    if event.key == pygame.K_t: self.pause_on_target    = not self.pause_on_target
                    if event.key == pygame.K_c: self.pause_on_collision = not self.pause_on_collision

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                if event.type == pygame.MOUSEMOTION and self._dragging_speed:
                    self._update_speed_from_mouse(event.pos[0])
                if event.type == pygame.MOUSEBUTTONUP:
                    self._dragging_speed = False

            self._render(scrub=(self._scrub is not None))
            self._clock.tick(30)

    def _handle_click(self, mouse_pos):
        px, py = mouse_pos[0] - _ENV_SIZE, mouse_pos[1]
        if self._btn_target    and self._btn_target.collidepoint(px, py):
            self.pause_on_target    = not self.pause_on_target
        if self._btn_collision and self._btn_collision.collidepoint(px, py):
            self.pause_on_collision = not self.pause_on_collision
        if self._slider_rect and self._slider_rect.collidepoint(*mouse_pos):
            self._dragging_speed = True
            self._update_speed_from_mouse(mouse_pos[0])

    def _update_speed_from_mouse(self, screen_x):
        if self._slider_rect is None:
            return
        ratio = max(0.0, min(1.0, (screen_x - self._slider_rect.x) / self._slider_rect.width))
        self._speed = max(1, int(round(1 + ratio * 59)))

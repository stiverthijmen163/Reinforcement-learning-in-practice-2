"""
Live debug viewer for the RL environment.

Active when no_gui=False. Creates a pygame window and renders the environment
at each training step. Use it as a drop-in wrapper around Environment:

    env = DebugViewer(env)   # wraps env; no-op when no_gui=True

Controls:
  Space      — pause / resume
  T          — toggle "pause on target reached"
  C          — toggle "pause on collision"
  ← →        — scrub through step history while paused
  Click HUD  — same as T / C toggles
"""
import collections
import pygame

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
        result = self.env.step(action)
        _, reward, done, info = result

        self._cum_r += reward
        self._step  += 1
        self._path.append(self.env.agent_pos)

        self._history.append({
            'pos':     self.env.agent_pos,
            'episode': self._episode,
            'step':    self._step,
            'cum_r':   self._cum_r,
            'reward':  reward,
            'done':    done,
            'info':    info,
        })

        if self.active:
            self._render()
            self._process_events()
            if self.pause_on_target    and info.get('target_reached'): self._paused = True
            if self.pause_on_collision and info.get('collided'):        self._paused = True
            if self._paused:
                self._wait_for_input()

        return result

    def update_metrics(self, epsilon=None, loss=None):
        """Call after each training step to update the HUD values."""
        if epsilon is not None: self.epsilon = epsilon
        if loss    is not None: self.loss    = loss

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

        # agent
        apos = self._history[self._scrub]['pos'] if (scrub and self._scrub is not None) else env.agent_pos
        if apos:
            ar   = max(3, int(env.agent_radius / env.x_max * _ENV_SIZE))
            col  = _SCRUB_DOT if (scrub and self._scrub is not None) else _AGENT
            pygame.draw.circle(surf, col, self._to_screen(*apos), ar)

        # border
        pygame.draw.rect(surf, (0, 0, 0), (0, 0, _ENV_SIZE, _ENV_SIZE), 2)

        self._draw_hud(scrub)
        pygame.display.flip()
        self._clock.tick(60)

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
        else:
            line("  PAUSED  " if self._paused else "  RUNNING ", _PAUSE_CLR if self._paused else _RUN_CLR)
            line(f"Ep    {self._episode}")
            line(f"Step  {self._step}")
            line(f"CumR  {self._cum_r:.1f}")
            if self.epsilon is not None: line(f"Eps   {self.epsilon:.4f}")
            if self.loss    is not None: line(f"Loss  {self.loss:.6f}")

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

        self._screen.blit(panel, (_ENV_SIZE, 0))

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
                            # step forward one live step — return control to training loop
                            self._paused = False
                            self._scrub  = None
                            self._render()
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

            self._render(scrub=(self._scrub is not None))
            self._clock.tick(30)

    def _handle_click(self, mouse_pos):
        px, py = mouse_pos[0] - _ENV_SIZE, mouse_pos[1]
        if self._btn_target    and self._btn_target.collidepoint(px, py):
            self.pause_on_target    = not self.pause_on_target
        if self._btn_collision and self._btn_collision.collidepoint(px, py):
            self.pause_on_collision = not self.pause_on_collision

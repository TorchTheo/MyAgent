"""Plugin manager with hot plugging.

- ``load(plugin)`` / ``unload(name)`` at any time, including while agents run
  (hooks iterate over an immutable snapshot, tools are re-resolved every step).
- Loading a plugin whose ``name`` is already loaded replaces it (hot swap).
- ``load_path(file)`` imports plugins from a Python file; ``watch(dir)`` polls a
  directory and (re)loads / unloads plugins as files are added, edited or deleted.
- ``load_entry_points()`` loads installed packages exposing ``agentharness.plugins``.
"""

from __future__ import annotations

import inspect
import sys
import threading
import traceback
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agentharness.core.events import LOG, Event
from agentharness.plugins.base import Plugin

if TYPE_CHECKING:
    from agentharness.core.runtime import Runtime


class PluginManager:
    def __init__(self, runtime: "Runtime"):
        self.runtime = runtime
        self._plugins: tuple[Plugin, ...] = ()
        self._sources: dict[str, list[str]] = {}  # file path -> plugin names loaded from it
        self._lock = threading.RLock()
        self._watchers: list[threading.Event] = []
        self._in_emit = threading.local()

    # ---- registry ------------------------------------------------------
    def load(self, plugin: Plugin) -> Plugin:
        if isinstance(plugin, type) and issubclass(plugin, Plugin):
            plugin = plugin()
        if not isinstance(plugin, Plugin):
            raise TypeError(f"expected Plugin, got {type(plugin).__name__}")
        with self._lock:
            if self.get(plugin.name):
                self.unload(plugin.name)
            tools = plugin.tools() or []
            registered: list[str] = []
            try:
                for t in tools:
                    self.runtime.tools.register(t, owner=f"plugin:{plugin.name}")
                    registered.append(t.name)
                plugin.on_load(self.runtime)
            except BaseException:
                for n in registered:
                    self.runtime.tools.unregister(n)
                raise
            self._plugins = self._plugins + (plugin,)
        return plugin

    def unload(self, name: str) -> Plugin | None:
        with self._lock:
            plugin = self.get(name)
            if plugin is None:
                return None
            self._plugins = tuple(p for p in self._plugins if p is not plugin)
            self.runtime.tools.unregister_owner(f"plugin:{name}")
        try:
            plugin.on_unload(self.runtime)
        except Exception:
            self._report(plugin, "on_unload")
        return plugin

    def get(self, name: str) -> Plugin | None:
        return next((p for p in self._plugins if p.name == name), None)

    def list(self) -> list[Plugin]:
        return list(self._plugins)

    def names(self) -> list[str]:
        return [p.name for p in self._plugins]

    # ---- dynamic loading ----------------------------------------------
    def load_path(self, path: str | Path) -> list[Plugin]:
        """Import a .py file and load the plugins it defines.

        The module may expose ``plugins = [...]``, ``plugin = ...``, a ``setup(runtime)``
        function returning plugin(s), or simply define ``Plugin`` subclasses.
        Reloading the same file replaces what it previously loaded; if the new version
        fails to import, the previous version stays loaded.
        """
        path = Path(path).resolve()
        with self._lock:
            mod_name = f"agentharness_plugin_{abs(hash(str(path)))}"
            # Compile from source on every load. The regular import path consults __pycache__,
            # whose staleness check (whole-second mtime + size) can serve the old bytecode when
            # a file is edited quickly without changing its size, silently defeating hot reload.
            source = path.read_bytes()
            module = types.ModuleType(mod_name)
            module.__file__ = str(path)
            sys.modules[mod_name] = module
            try:
                exec(compile(source, str(path), "exec"), module.__dict__)
            except BaseException:
                raise  # previous version (if any) stays loaded
            new_plugins = self._discover(module)
            # swap only after the new version compiled and was discovered successfully
            for name in self._sources.pop(str(path), []):
                self.unload(name)
            loaded = [self.load(p) for p in new_plugins]
            self._sources[str(path)] = [p.name for p in loaded]
            return loaded

    def _discover(self, module: Any) -> list[Plugin]:
        if hasattr(module, "setup") and callable(module.setup):
            found = module.setup(self.runtime)
        elif hasattr(module, "plugins"):
            found = module.plugins
        elif hasattr(module, "plugin"):
            found = module.plugin
        else:
            found = [obj for obj in vars(module).values()
                     if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin
                     and obj.__module__ == module.__name__]
        if found is None:
            return []
        items = found if isinstance(found, (list, tuple)) else [found]
        return [p() if isinstance(p, type) else p for p in items]

    def load_entry_points(self, group: str = "agentharness.plugins") -> list[Plugin]:
        from importlib.metadata import entry_points

        loaded = []
        for ep in entry_points(group=group):
            obj = ep.load()
            items = obj(self.runtime) if inspect.isfunction(obj) else obj
            for p in items if isinstance(items, (list, tuple)) else [items]:
                loaded.append(self.load(p))
        return loaded

    def watch(self, directory: str | Path, interval: float = 1.0,
              on_change: Callable[[str, Path, BaseException | None], None] | None = None) -> threading.Event:
        """Hot-reload every ``*.py`` in ``directory`` (files starting with '_' are ignored).

        Returns an Event; ``set()`` it to stop watching. ``on_change(action, path, error)``
        is called with action in {'load', 'reload', 'unload', 'error'}.
        """
        directory = Path(directory).resolve()
        stop = threading.Event()
        mtimes: dict[Path, tuple[int, int]] = {}

        def notify(action: str, p: Path, err: BaseException | None = None) -> None:
            self._log(f"plugin {action}: {p.name}", path=str(p), error=repr(err) if err else None)
            if on_change:
                on_change(action, p, err)

        def scan() -> None:
            current = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in directory.glob("*.py") if not p.name.startswith("_")}
            for p in list(mtimes):
                if p not in current:
                    del mtimes[p]
                    with self._lock:
                        for name in self._sources.pop(str(p), []):
                            self.unload(name)
                    notify("unload", p)
            for p, mt in sorted(current.items()):
                if mtimes.get(p) == mt:
                    continue
                action = "reload" if p in mtimes else "load"
                mtimes[p] = mt
                try:
                    self.load_path(p)
                    notify(action, p)
                except Exception as e:  # keep watching; a broken edit shouldn't kill the runtime
                    notify("error", p, e)

        directory.mkdir(parents=True, exist_ok=True)
        scan()

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    scan()
                except Exception:
                    traceback.print_exc()

        threading.Thread(target=loop, name=f"plugin-watch:{directory.name}", daemon=True).start()
        self._watchers.append(stop)
        return stop

    def stop_watching(self) -> None:
        for s in self._watchers:
            s.set()
        self._watchers.clear()

    # ---- dispatch -----------------------------------------------------
    def emit(self, event: Event) -> None:
        for p in self._plugins:
            try:
                p.on_event(event)
            except Exception:
                if p.strict:
                    raise
                # don't recurse into emit while reporting an emit failure
                traceback.print_exc()

    def call(self, hook: str, *args: Any) -> list[tuple[Plugin, Any]]:
        """Call ``hook`` on every plugin; returns (plugin, non-None result) pairs."""
        out = []
        for p in self._plugins:
            fn = getattr(p, hook, None)
            if fn is None:
                continue
            try:
                r = fn(*args)
            except Exception:
                if p.strict:
                    raise
                self._report(p, hook)
                continue
            if r is not None:
                out.append((p, r))
        return out

    def _report(self, plugin: Plugin, hook: str) -> None:
        self._log(f"plugin {plugin.name}.{hook} failed", traceback=traceback.format_exc(limit=8))

    def _log(self, message: str, **data: Any) -> None:
        from agentharness.core.tracing import current_span
        from agentharness.core.messages import new_id

        sp = current_span()
        self.emit(Event(LOG, sp.trace_id if sp else new_id("tr_"), sp.id if sp else None,
                        sp.parent_id if sp else None, "log", message, {"message": message, **data}))

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random

import dearpygui.dearpygui as dpg

from simulation.engine import SimulationEngine
from simulation.environment import LineagePreset, SimulationConfig
from simulation.replicator import Replicator
from analysis.metrics import MetricsTracker


@dataclass
class VisualParticle:
    x: float
    y: float
    vx: float
    vy: float


class DpgUI:
    def __init__(
        self,
        engine: SimulationEngine,
        metrics: MetricsTracker,
        config: SimulationConfig,
    ) -> None:
        self.engine = engine
        self.metrics = metrics
        self.config = config
        self.running = False
        self.step_accumulator = 0.0
        self._run_until_step: int | None = None
        self._run_until_input_tag = "run_until_input"
        self.particles: Dict[int, VisualParticle] = {}
        self.drawlist_id: int | None = None
        self.plot_series: Dict[str, int] = {}
        self._series_sources: Dict[str, str] = {}
        self._plot_x_axes: Dict[str, int] = {}
        self._plot_y_axes: Dict[str, int] = {}
        self._lineage_names = list(self.metrics.lineage_names)
        self._trait_classes = list(self.metrics.trait_classes)
        self._plot_groups = {
            "population": ["population", "diversity", "resources"],
            "traits": ["avg_replication_rate", "avg_fidelity", "avg_stability"],
            "trait_classes": [f"trait::{name}" for name in self._trait_classes],
        }
        self._lineage_palette = [
            (231, 76, 60, 255),
            (41, 128, 185, 255),
            (39, 174, 96, 255),
            (142, 68, 173, 255),
            (243, 156, 18, 255),
            (44, 62, 80, 255),
        ]
        self._lineage_colors = {
            "Balanced": (231, 76, 60, 255),
            "High Stability": (39, 174, 96, 255),
            "High Fidelity": (243, 156, 18, 255),
            "High Replication": (41, 128, 185, 255),
        }
        self._lineage_display = list(self._lineage_colors.keys())
        self._steps_per_second = max(1.0, 1.0 / max(self.config.step_interval, 0.001))
        self._visual_rng = random.Random(1337)
        self._draw_width = 600
        self._draw_height = 400

    def run(self) -> None:
        dpg.create_context()
        self._build_ui()
        dpg.create_viewport(
            title="Primordial Soup Replicator Simulation",
            width=1920,
            height=1080,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()
        self.metrics.update(self.engine.step_count, self.engine.environment)

        while dpg.is_dearpygui_running():
            dt = dpg.get_delta_time()
            self._update(dt)
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _build_ui(self) -> None:
        with dpg.window(label="Controls", width=320, height=700, pos=(10, 10)):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start", callback=self._toggle_running, tag="start_btn"
                )
                dpg.add_button(label="Step", callback=self._step_once)
                dpg.add_button(label="Reset", callback=self._reset)

            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Run Until",
                    default_value=200,
                    min_value=0,
                    min_clamped=True,
                    tag=self._run_until_input_tag,
                    width=120,
                )
                dpg.add_button(label="Run To", callback=self._run_until)

            dpg.add_separator()
            dpg.add_text("Simulation")
            dpg.add_slider_float(
                label="Replication Mult",
                min_value=0.0,
                max_value=3.0,
                default_value=self.config.replication_multiplier,
                callback=self._on_replication_multiplier,
            )
            dpg.add_slider_float(
                label="Mutation Mult",
                min_value=0.0,
                max_value=3.0,
                default_value=self.config.mutation_multiplier,
                callback=self._on_mutation_multiplier,
            )
            dpg.add_slider_float(
                label="Stability Mult",
                min_value=0.2,
                max_value=1.2,
                default_value=self.config.stability_multiplier,
                callback=self._on_stability_multiplier,
            )
            dpg.add_slider_float(
                label="Stability Cap",
                min_value=0.6,
                max_value=1.0,
                default_value=self.config.stability_cap,
                format="%.3f",
                callback=self._on_stability_cap,
            )
            dpg.add_slider_float(
                label="Steps / Sec",
                min_value=1,
                max_value=120.0,
                default_value=self._steps_per_second,
                format="%.0f sps",
                callback=self._on_steps_per_second,
            )
            dpg.add_slider_int(
                label="Resource Regen",
                min_value=0,
                max_value=200,
                default_value=self.config.resource_regen_rate,
                callback=self._on_resource_regen,
            )

            dpg.add_separator()
            with dpg.collapsing_header(label="Lineage Setup", default_open=False):
                dpg.add_text("Changes apply after Reset.")
                dpg.add_slider_float(
                    label="Trait Jitter",
                    min_value=0.0,
                    max_value=0.05,
                    default_value=self.config.lineage_trait_jitter,
                    format="%.3f",
                    callback=self._on_lineage_jitter,
                )
                dpg.add_slider_float(
                    label="Balance Threshold",
                    min_value=0.0,
                    max_value=0.3,
                    default_value=self.config.trait_class_balance_threshold,
                    format="%.3f",
                    callback=self._on_trait_balance_threshold,
                )
                dpg.add_text("Weights are relative; ratios matter.")
                for index, preset in enumerate(self.config.lineage_presets):
                    with dpg.tree_node(label=preset.name, default_open=False):
                        dpg.add_slider_float(
                            label="Replication Rate",
                            min_value=0.0,
                            max_value=0.2,
                            default_value=preset.replication_rate,
                            format="%.3f",
                            callback=self._on_lineage_value,
                            user_data=(index, "replication_rate"),
                        )
                        dpg.add_slider_float(
                            label="Fidelity",
                            min_value=0.5,
                            max_value=1.0,
                            default_value=preset.fidelity,
                            format="%.3f",
                            callback=self._on_lineage_value,
                            user_data=(index, "fidelity"),
                        )
                        dpg.add_slider_float(
                            label="Stability",
                            min_value=0.5,
                            max_value=1.0,
                            default_value=preset.stability,
                            format="%.3f",
                            callback=self._on_lineage_value,
                            user_data=(index, "stability"),
                        )
                        dpg.add_slider_float(
                            label="Weight",
                            min_value=0.0,
                            max_value=1.0,
                            default_value=preset.weight,
                            format="%.2f",
                            callback=self._on_lineage_value,
                            user_data=(index, "weight"),
                        )
                dpg.add_button(label="Apply & Reset", callback=self._reset)

            dpg.add_separator()
            dpg.add_text("Stats")
            dpg.add_text("Step: 0", tag="stat_step")
            dpg.add_text("Population: 0", tag="stat_population")
            dpg.add_text("Resources: 0", tag="stat_resources")
            dpg.add_text("Diversity: 0", tag="stat_diversity")
            dpg.add_text("Dominant Lineage: -", tag="stat_dominant_lineage")
            dpg.add_text("Dominant Trait Class: -", tag="stat_dominant_trait")

        with dpg.window(
            label="Primordial Soup",
            width=self._draw_width + 40,
            height=self._draw_height + 60,
            pos=(350, 10),
        ):
            self.drawlist_id = dpg.add_drawlist(
                width=self._draw_width, height=self._draw_height
            )

        with dpg.window(label="Metrics", width=500, height=700, pos=(1000, 10)):
            with dpg.tab_bar():
                with dpg.tab(label="Population & Traits"):
                    with dpg.plot(label="Population", height=300, width=480):
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Count")
                        self._plot_x_axes["population"] = x_axis
                        self._plot_y_axes["population"] = y_axis
                        self.plot_series["population"] = dpg.add_line_series(
                            [], [], label="Population", parent=y_axis
                        )
                        self._series_sources["population"] = "population"
                        self.plot_series["diversity"] = dpg.add_line_series(
                            [], [], label="Diversity", parent=y_axis
                        )
                        self._series_sources["diversity"] = "diversity"
                        self.plot_series["resources"] = dpg.add_line_series(
                            [], [], label="Resources", parent=y_axis
                        )
                        self._series_sources["resources"] = "resources"

                    with dpg.plot(label="Traits", height=300, width=480):
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Average")
                        self._plot_x_axes["traits"] = x_axis
                        self._plot_y_axes["traits"] = y_axis
                        self.plot_series["avg_replication_rate"] = dpg.add_line_series(
                            [], [], label="Replication Rate", parent=y_axis
                        )
                        self._series_sources["avg_replication_rate"] = (
                            "avg_replication_rate"
                        )
                        self.plot_series["avg_fidelity"] = dpg.add_line_series(
                            [], [], label="Fidelity", parent=y_axis
                        )
                        self._series_sources["avg_fidelity"] = "avg_fidelity"
                        self.plot_series["avg_stability"] = dpg.add_line_series(
                            [], [], label="Stability", parent=y_axis
                        )
                        self._series_sources["avg_stability"] = "avg_stability"

                with dpg.tab(label="Lineages"):
                    dpg.add_text("Lineage Averages (rep/fid/stab | n)")
                    for name in self._lineage_display:
                        dpg.add_text(
                            f"{name}: rep 0.000 | fid 0.000 | stab 0.000 | n=0",
                            tag=self._lineage_avg_tag(name),
                            color=self._lineage_colors.get(name),
                        )

                    dpg.add_spacer(height=10)

                    with dpg.plot(label="Trait Classes", height=300, width=480):
                        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Count")
                        self._plot_x_axes["trait_classes"] = x_axis
                        self._plot_y_axes["trait_classes"] = y_axis
                        for name in self._trait_classes:
                            key = f"trait::{name}"
                            self.plot_series[key] = dpg.add_line_series(
                                [], [], label=name, parent=y_axis
                            )
                            self._series_sources[key] = name

    def _update(self, dt: float) -> None:
        if self.running:
            self.step_accumulator += dt
            interval = 1.0 / max(self._steps_per_second, 1.0)
            while self.step_accumulator >= interval:
                self.engine.step()
                self.metrics.update(self.engine.step_count, self.engine.environment)
                self.step_accumulator -= interval
                if (
                    self._run_until_step is not None
                    and self.engine.step_count >= self._run_until_step
                ):
                    self.running = False
                    self._run_until_step = None
                    dpg.set_item_label("start_btn", "Start")
                    break

        self._sync_particles()
        self._update_particles(dt)
        self._draw_particles()
        self._update_stats()
        self._update_lineage_averages()
        self._update_plots()

    def _toggle_running(self, sender: int) -> None:
        self.running = not self.running
        label = "Pause" if self.running else "Start"
        dpg.set_item_label("start_btn", label)

    def _step_once(self, sender: int) -> None:
        if not self.running:
            self.engine.step()
            self.metrics.update(self.engine.step_count, self.engine.environment)

    def _reset(self, sender: int) -> None:
        self.running = False
        self._run_until_step = None
        dpg.set_item_label("start_btn", "Start")
        self.engine.reset()
        self.metrics.reconfigure(self.config)
        self.metrics.update(self.engine.step_count, self.engine.environment)
        self.particles.clear()

    def _run_until(self, sender: int) -> None:
        target = int(dpg.get_value(self._run_until_input_tag))
        if target <= self.engine.step_count:
            return
        self.running = True
        self._run_until_step = target
        dpg.set_item_label("start_btn", "Pause")

    def _on_replication_multiplier(self, sender: int, value: float) -> None:
        self.config.replication_multiplier = float(value)

    def _on_mutation_multiplier(self, sender: int, value: float) -> None:
        self.config.mutation_multiplier = float(value)

    def _on_stability_multiplier(self, sender: int, value: float) -> None:
        self.config.stability_multiplier = float(value)

    def _on_stability_cap(self, sender: int, value: float) -> None:
        self.config.stability_cap = float(value)

    def _on_steps_per_second(self, sender: int, value: float) -> None:
        self._steps_per_second = max(1.0, float(value))

    def _on_resource_regen(self, sender: int, value: int) -> None:
        self.config.resource_regen_rate = int(value)

    def _on_lineage_jitter(self, sender: int, value: float) -> None:
        self.config.lineage_trait_jitter = float(value)

    def _on_trait_balance_threshold(self, sender: int, value: float) -> None:
        self.config.trait_class_balance_threshold = float(value)

    def _on_lineage_value(self, sender: int, value: float, user_data: tuple) -> None:
        index, field = user_data
        presets = list(self.config.lineage_presets)
        if index < 0 or index >= len(presets):
            return
        preset = presets[index]
        updates = {
            "replication_rate": preset.replication_rate,
            "fidelity": preset.fidelity,
            "stability": preset.stability,
            "weight": preset.weight,
        }
        updates[field] = float(value)
        presets[index] = LineagePreset(
            name=preset.name,
            replication_rate=updates["replication_rate"],
            fidelity=updates["fidelity"],
            stability=updates["stability"],
            weight=updates["weight"],
        )
        self.config.lineage_presets = presets

    def _sync_particles(self) -> None:
        current = {rep.uid for rep in self.engine.environment.replicators}
        for uid in list(self.particles.keys()):
            if uid not in current:
                del self.particles[uid]
        for rep in self.engine.environment.replicators:
            if rep.uid not in self.particles:
                self.particles[rep.uid] = self._spawn_particle()

    def _spawn_particle(self) -> VisualParticle:
        x = self._visual_rng.uniform(10.0, self._draw_width - 10.0)
        y = self._visual_rng.uniform(10.0, self._draw_height - 10.0)
        vx = self._visual_rng.uniform(-10.0, 10.0)
        vy = self._visual_rng.uniform(-10.0, 10.0)
        return VisualParticle(x=x, y=y, vx=vx, vy=vy)

    def _update_particles(self, dt: float) -> None:
        jitter = 30.0
        max_speed = 40.0
        for particle in self.particles.values():
            particle.vx += self._visual_rng.uniform(-jitter, jitter) * dt
            particle.vy += self._visual_rng.uniform(-jitter, jitter) * dt
            particle.vx = max(-max_speed, min(max_speed, particle.vx))
            particle.vy = max(-max_speed, min(max_speed, particle.vy))
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt

            if particle.x < 8.0 or particle.x > self._draw_width - 8.0:
                particle.vx *= -1.0
                particle.x = max(8.0, min(self._draw_width - 8.0, particle.x))
            if particle.y < 8.0 or particle.y > self._draw_height - 8.0:
                particle.vy *= -1.0
                particle.y = max(8.0, min(self._draw_height - 8.0, particle.y))

    def _draw_particles(self) -> None:
        if self.drawlist_id is None:
            return
        dpg.delete_item(self.drawlist_id, children_only=True)
        for rep in self.engine.environment.replicators:
            particle = self.particles.get(rep.uid)
            if particle is None:
                continue
            radius = 2.5 + rep.length * 0.45
            color = self._color_from_lineage(rep)
            dpg.draw_circle(
                (particle.x, particle.y),
                radius,
                color=color,
                fill=color,
                parent=self.drawlist_id,
            )

    def _update_stats(self) -> None:
        dpg.set_value("stat_step", f"Step: {self.engine.step_count}")
        dpg.set_value(
            "stat_population", f"Population: {len(self.engine.environment.replicators)}"
        )
        dpg.set_value(
            "stat_resources", f"Resources: {self.engine.environment.resources}"
        )
        dpg.set_value(
            "stat_diversity",
            f"Diversity: {self.metrics.history['diversity'][-1] if self.metrics.history['diversity'] else 0}",
        )
        snapshot = self.metrics.last_snapshot
        if snapshot is None:
            dpg.set_value("stat_dominant_lineage", "Dominant Lineage: -")
            dpg.set_value("stat_dominant_trait", "Dominant Trait Class: -")
        else:
            dpg.set_value(
                "stat_dominant_lineage",
                f"Dominant Lineage: {snapshot.dominant_lineage}",
            )
            dpg.set_value(
                "stat_dominant_trait",
                f"Dominant Trait Class: {snapshot.dominant_trait_class}",
            )

    def _update_lineage_averages(self) -> None:
        averages = self.metrics.last_lineage_averages
        for name in self._lineage_display:
            rep, fid, stab, count = averages.get(name, (0.0, 0.0, 0.0, 0))
            dpg.set_value(
                self._lineage_avg_tag(name),
                f"{name}: rep {rep:.3f} | fid {fid:.3f} | stab {stab:.3f} | n={count}",
            )

    @staticmethod
    def _lineage_avg_tag(name: str) -> str:
        return f"lineage_avg::{name}"

    def _update_plots(self) -> None:
        steps = self.metrics.step_series()
        if not steps:
            steps = [1]
        plot_steps = [max(1, step) for step in steps]
        series_cache: Dict[str, List[float]] = {}
        for name, series_id in self.plot_series.items():
            series_name = self._series_sources.get(name, name)
            values = self.metrics.series(series_name)
            if not values:
                values = [0]
            series_cache[name] = values
            dpg.set_value(series_id, [plot_steps, values])

        max_step = max(plot_steps) if plot_steps else 1
        for axis_id in self._plot_x_axes.values():
            dpg.set_axis_limits(axis_id, 1, max_step)

        for plot_name, series_names in self._plot_groups.items():
            axis_id = self._plot_y_axes.get(plot_name)
            if axis_id is None:
                continue
            combined: List[float] = []
            for series_name in series_names:
                combined.extend(series_cache.get(series_name, []))
            if not combined:
                dpg.set_axis_limits(axis_id, 0, 1)
                continue
            min_val = min(combined)
            max_val = max(combined)
            if min_val == max_val:
                max_val = min_val + 1.0
            padding = (max_val - min_val) * 0.1
            min_val = max(0.0, min_val - padding)
            max_val += padding
            dpg.set_axis_limits(axis_id, min_val, max_val)

    def _color_from_lineage(self, rep: Replicator) -> Tuple[int, int, int, int]:
        if rep.lineage_name in self._lineage_colors:
            base = self._lineage_colors[rep.lineage_name]
        elif rep.lineage_name in self._lineage_names:
            index = self._lineage_names.index(rep.lineage_name)
            base = self._lineage_palette[index % len(self._lineage_palette)]
        else:
            base = (160, 160, 160, 255)
        alpha = int(90 + 165 * rep.stability)
        return base[0], base[1], base[2], alpha

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        if s == 0.0:
            return v, v, v
        h = (h % 1.0) * 6.0
        i = int(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        if i == 0:
            return v, t, p
        if i == 1:
            return q, v, p
        if i == 2:
            return p, v, t
        if i == 3:
            return p, q, v
        if i == 4:
            return t, p, v
        return v, p, q

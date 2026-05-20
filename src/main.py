from simulation.environment import SimulationConfig
from simulation.engine import SimulationEngine
from analysis.metrics import MetricsTracker
from ui.dpg_ui import DpgUI


def main() -> None:
    config = SimulationConfig()
    engine = SimulationEngine(config)
    metrics = MetricsTracker(config, max_history=300)
    ui = DpgUI(engine, metrics, config)
    ui.run()


if __name__ == "__main__":
    main()

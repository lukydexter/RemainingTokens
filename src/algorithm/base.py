from abc import ABC, abstractmethod
from ..monitoring import MonitoringConfig, monitor_performance


class Algorithm(ABC):
    """Base class for algorithms with optional monitoring configuration."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        execute = cls.__dict__.get("execute")
        if execute is None:
            return

        if getattr(execute, "__isabstractmethod__", False):
            return

        if callable(execute) and not getattr(execute, "__monitor_wrapped__", False):
            cls.execute = monitor_performance(execute)

    def __init__(self, monitoring: MonitoringConfig | None = None):
        self.monitoring = monitoring or MonitoringConfig.from_file()
        self.analytics = self.monitoring.analytics

    @abstractmethod
    @monitor_performance
    def execute(self):
        pass

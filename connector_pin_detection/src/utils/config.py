"""配置管理模块"""
import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default_config.yaml")


class Config:
    """全局配置类"""

    def __init__(self, config_path=None):
        self._data = self._load_default()
        if config_path and os.path.exists(config_path):
            user_cfg = self._load_yaml(config_path)
            self._deep_merge(self._data, user_cfg)

    @staticmethod
    def _load_default():
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default_config.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base, overlay):
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                Config._deep_merge(base[k], v)
            else:
                base[k] = v

    def get(self, *keys, default=None):
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def raw(self):
        return self._data

    def save(self):
        """保存当前配置到默认配置文件的路径"""
        path = DEFAULT_CONFIG_PATH
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return True

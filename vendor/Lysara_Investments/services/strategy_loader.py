import importlib
import pathlib
from types import ModuleType
from typing import Dict, List

class StrategyLoader:
    """Hot loads strategy modules from a folder."""

    def __init__(self, folder: str = "strategies"):
        self.folder = pathlib.Path(folder)
        self.modules: Dict[str, ModuleType] = {}

    def load_strategies(self) -> Dict[str, ModuleType]:
        for file in self.folder.glob("*.py"):
            mod_name = file.stem
            spec = importlib.util.spec_from_file_location(mod_name, file)
            if not spec or not spec.loader:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[mod_name] = module
        return self.modules

    def reload(self, mod_name: str) -> ModuleType | None:
        module = self.modules.get(mod_name)
        if module:
            return importlib.reload(module)
        return None

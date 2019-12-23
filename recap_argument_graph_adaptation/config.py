import collections
from pathlib import Path
from typing import Dict, Any

import tomlkit as toml


class Config(collections.MutableMapping):
    _instance = None
    _store: Dict[str, Any]
    _file = Path("config.toml")
    _template = Path("config-template.toml")

    @classmethod
    def instance(cls):
        """ Static access method. """
        if cls._instance == None:
            cls()
        return cls._instance

    def __init__(self):
        """ Private constructor."""
        if Config._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Config._instance = self
            with self._file.open() as f:
                self._store = toml.parse(f.read())

    def __getitem__(self, key):
        if key in self._store:
            return self._store[key]
        else:
            raise ValueError(self._error(key))

    def __setitem__(self, key, value):
        if key in self._store:
            self._store[key] = value
        else:
            raise ValueError(self._error(key))

    def __delitem__(self, key):
        if key in self._store:
            del self._store[key]
        else:
            raise ValueError(self._error(key))

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return repr(self._store)

    def __str__(self):
        return str(self._store)

    def _error(self, key: str) -> str:
        return f"""The key '{key}' is not defined in '{self._file}'.
        This is most likely caused by an old version of that file.
        Look at '{self._template}' to see all options."""


config = Config.instance()

import collections
from pathlib import Path
import typing as t

import tomlkit as toml


class Config(collections.MutableMapping):
    _instance = None
    _store: t.MutableMapping[str, t.Any]
    _file = Path("config.toml")

    @classmethod
    def instance(cls):
        """ Static access method. """
        if cls._instance is None:
            cls()
        return cls._instance

    def __init__(self):
        """ Private constructor."""
        if Config._instance is not None:
            raise RuntimeError("This class is a singleton!")
        else:
            Config._instance = self
            with self._file.open() as f:
                self._store = toml.parse(f.read())

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return repr(self._store)

    def __str__(self):
        return str(self._store)


config = Config.instance()

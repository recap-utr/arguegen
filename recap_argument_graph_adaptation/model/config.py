from __future__ import annotations

import collections
from pathlib import Path
import typing as t
from packaging import version

import tomlkit as toml


class Config(collections.MutableMapping):
    _instance = None
    _store: t.MutableMapping[str, t.Any]
    _file = Path("config.toml")
    _template = Path("config-example.toml")

    @classmethod
    def instance(cls) -> Config:
        """ Static access method. """
        return cls._instance or cls()

    def __init__(self):
        """ Private constructor."""
        if Config._instance is not None:
            raise RuntimeError("This class is a singleton!")
        else:
            Config._instance = self
            with self._file.open() as f:
                self._store = toml.parse(f.read())

            with self._template.open() as f:
                template = toml.parse(f.read())

            if "version" not in self:
                raise RuntimeError(
                    f"The attribute 'version' is missing in '{self._file}'."
                )

            if "version" not in template:
                raise RuntimeError(
                    f"The attribute 'version' is missing in '{self._template}'."
                )

            self_version = version.parse(str(self["version"]))
            template_version = version.parse(str(template["version"]))

            if self_version != template_version:
                raise RuntimeError(
                    f"The version of '{self._file}' ({self_version}) is not equal to "
                    f"the version of '{self._template}' ({template_version}). "
                    "Please update your config!"
                )

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

    def tuning(self, prefix: str, name: t.Optional[str] = None) -> t.Any:
        prefix = f"{prefix}_"
        store = self._store["_tuning"]

        if name:
            return store[prefix + name]

        return {
            key[len(prefix) :]: value
            for key, value in store.items()
            if key.startswith(prefix)
        }


config = Config.instance()

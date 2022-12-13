from typing import Any, Mapping, Optional

from dynaconf import Dynaconf

config: Any = Dynaconf(
    envvar_prefix="CONFIG",
    settings_files=["settings.toml", ".secrets.toml"],
)


def tuning_run(obj: Mapping[str, Any]) -> bool:
    return "_tuning" in obj


def tuning(
    obj: Mapping[str, Any],
    prefix: Optional[str] = None,
    name: Optional[str] = None,
    postfix_overwrite: Optional[str] = None,
) -> Any:
    mapping = obj["_tuning"]

    if prefix:
        if name and postfix_overwrite:
            result = mapping.get("_".join([prefix, name, postfix_overwrite]))

            if result is not None:
                return result

        if name:
            return mapping["_".join([prefix, name])]

        return {
            key[len(prefix) + 1 :]: value
            for key, value in mapping.items()
            if key.startswith(f"{prefix}_")
        }

    return mapping


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

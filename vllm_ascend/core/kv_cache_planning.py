# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Select the upstream planning API without changing the upstream checkout."""

import importlib
import importlib.util
import sys

from vllm.v1.core import kv_cache_utils as legacy_planning

try:
    from vllm.v1.core import kv_cache_planning as planning
except ImportError:
    from vllm.v1.core import kv_cache_utils as planning


def install_legacy_planning_exports() -> None:
    """Keep Qwen startup's shared imports working after Core split planning.

    Re-export the actual upstream helpers, only when their old names are
    absent. This does not replace implementations or change cache semantics.
    Remove once the shared startup imports no longer require the old module.
    """
    if planning is legacy_planning:
        return
    for name in ("may_override_num_blocks", "create_kv_cache_group_specs"):
        if not hasattr(legacy_planning, name):
            setattr(legacy_planning, name, getattr(planning, name))


def install_qwen_startup_config_import() -> None:
    """Resolve a removed import used transitively by Qwen's shared runner.

    The reference upstream now uses the Transformers configuration module.
    Reuse that real module, not a stub or a replacement model implementation.
    Preserve an installed legacy module and never modify the checkpoint.
    """
    if planning is legacy_planning:
        return
    legacy = "vllm.transformers_utils.configs.deepseek_v4"
    if legacy in sys.modules or importlib.util.find_spec(legacy) is not None:
        return
    provider = importlib.import_module("transformers.models.deepseek_v4.configuration_deepseek_v4")
    sys.modules[legacy] = provider


install_legacy_planning_exports()
install_qwen_startup_config_import()

_approximate_gcd = planning._approximate_gcd
may_override_num_blocks = planning.may_override_num_blocks
create_kv_cache_group_specs = planning.create_kv_cache_group_specs


def get_kv_cache_groups(*args, **kwargs):
    # Resolve after platform patches have been installed.
    return planning.get_kv_cache_groups(*args, **kwargs)


def unify_hybrid_kv_cache_specs(*args, **kwargs):
    return planning.unify_hybrid_kv_cache_specs(*args, **kwargs)

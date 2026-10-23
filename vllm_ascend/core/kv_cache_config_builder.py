# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend placement on the upstream Core-owned PP planning flow."""

from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def _get_pool_bytes_per_block(self, vllm_config, kv_cache_groups):
        from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_pool_bytes_per_block

        if not kv_cache_groups:
            return None
        return _ascend_pool_bytes_per_block(vllm_config, kv_cache_groups)

    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
        from vllm_ascend.patch.platform.patch_kv_cache_utils import (
            _get_qwen4_exp_six_region_kv_cache_config,
        )

        # Core has already projected the global groups to this PP worker and
        # converged the block count. Do not independently fit it a second time.
        config = _get_qwen4_exp_six_region_kv_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
        if config is None:
            config = super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
        config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        return config

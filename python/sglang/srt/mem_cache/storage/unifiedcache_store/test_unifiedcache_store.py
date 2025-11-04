import pytest
from unittest.mock import patch
import threading
import torch
import random
import os
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost, MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.unifiedcache_store.unifiedcache_store import UnifiedCacheStore


@pytest.fixture(
    params=[1, 4], ids=lambda v: f"TP_size{v}", scope="function"
)
def tp_size(request):
    return request.param


@pytest.fixture(
    params=[True], ids=lambda v: "page_first", scope="function"
)
def page_first(request):
    return request.param


def init_mem_pool_host(is_mla):
    size = 1024
    page_size = 128
    dtype = torch.float16
    layer_num = 2
    device_l1 = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
    enable_memory_saver = False
    if is_mla:
        kv_lora_rank = 8
        qk_rope_head_dim = 128
        device_pool = MLATokenToKVPool(size, page_size, dtype, kv_lora_rank, qk_rope_head_dim, layer_num, device_l1, enable_memory_saver)
    else:
        head_num = 4
        head_dim = 8
        device_pool = MHATokenToKVPool(size, page_size, dtype, head_num, head_dim, layer_num, device_l1, enable_memory_saver)

    host_to_device_ratio = 1.0
    host_size = 1
    host_page_size = page_size
    layout = "page_first"
    pin_memory = True

    if is_mla:
        pool_host = MLATokenToKVPoolHost(device_pool, host_to_device_ratio, host_size, host_page_size, layout, pin_memory)
    else:
        pool_host = MHATokenToKVPoolHost(device_pool, host_to_device_ratio, host_size, host_page_size, layout, pin_memory)
    return pool_host


def string_to_uint64(key_str: str) -> str:
    key_bytes = key_str.encode('ascii')
    padded_bytes = key_bytes.ljust(16, b'\0')[:16]
    result_str = padded_bytes.decode('ascii', errors='ignore')
    return result_str


def generate_data_and_get_pool_host(is_mla):
    mem_pool_host = init_mem_pool_host(is_mla)
    num_pages = 4
    for i in range(num_pages):
        if is_mla:
            data_page = torch.randn(mem_pool_host.device_pool.page_size
                                    * mem_pool_host.device_pool.layer_num
                                    * (mem_pool_host.device_pool.kv_lora_rank + mem_pool_host.device_pool.qk_rope_head_dim),
                                    dtype=mem_pool_host.device_pool.dtype)
        else:
            data_page = torch.randn(2 * mem_pool_host.device_pool.layer_num
                                    * mem_pool_host.device_pool.page_size
                                    * mem_pool_host.device_pool.head_num
                                    * mem_pool_host.device_pool.head_dim,
                                    dtype=mem_pool_host.device_pool.dtype)
        mem_pool_host.set_from_flat_data_page(index=i * mem_pool_host.device_pool.page_size, data_page=data_page)

    keys = [string_to_uint64(f"test_key_{i}_{random.randint(1000, 9999)}") for i in range(num_pages)]
    host_indices = torch.tensor(
        [i for i in range(num_pages * mem_pool_host.device_pool.page_size)]
    )
    return keys, host_indices, mem_pool_host


_current_tp_size = threading.local()


@pytest.fixture(autouse=True, scope="function")
def _set_current_tp_size(tp_size):
    _current_tp_size.value = tp_size


@pytest.fixture(autouse=True, scope="session")
def _patch_dist():
    def fake_new_group(ranks, backend=None):
        return None

    def fake_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None):
        tensor.fill_(getattr(_current_tp_size, "value", 1))

    def fake_get_process_group_ranks(group):
        size = getattr(_current_tp_size, "value", 1)
        return list(range(size))

    with patch("torch.distributed.new_group", side_effect=fake_new_group), \
         patch("torch.distributed.all_reduce", side_effect=fake_all_reduce), \
         patch("torch.distributed.get_process_group_ranks",
               side_effect=fake_get_process_group_ranks):
        yield


# ============================================================================
# UnifiedCache Integration Tests
# ============================================================================


def test_batch_operation_mla(tp_size, page_first):
    """MLA test the batch set/get APIs with multiple key-value pairs"""
    print("=" * 100)

    keys, host_indices, mem_pool_host = generate_data_and_get_pool_host(is_mla=True)
    test_dir = "/home/ucm_storage_data_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for i in range(tp_size):
        config = HiCacheStorageConfig(
            tp_rank=i,
            tp_size=tp_size,
            is_mla_model=True,
            is_page_first_layout=page_first,
            model_name="test",
            extra_config = {
                "kv_connector_extra_config": {
                    "ucm_connector_name": "UcmNfsStore",
                    "ucm_connector_config": {
                        "storage_backends": test_dir,
                        "max_cache_size": 10240,
                        "kv_block_size": 262144
                    }
                }
            }
        )

        store = UnifiedCacheStore(
            storage_config=config, mem_pool_host=mem_pool_host, tp_group=None)
        store.register_mem_pool_host(mem_pool_host)

        if i == 0:
            exist_result = store.batch_exists(keys)
            expected = 0
            assert exist_result == expected

            set_results = store.batch_set_v1(keys, host_indices)
            expected = [True] * len(set_results)
            assert set_results == expected
        
        kv_buffer = mem_pool_host.kv_buffer
        for j in range(len(host_indices)):
            mem_pool_host.kv_buffer[host_indices[j]].zero_()
            assert torch.all(mem_pool_host.kv_buffer[host_indices[j]] == 0)
        
        exist_result = store.batch_exists(keys)
        expected = len(keys)
        assert exist_result == expected

        get_results = store.batch_get_v1(keys, host_indices)
        expected = [True] * len(get_results)
        assert get_results == expected

        for j in range(len(host_indices)):
            torch.equal(mem_pool_host.kv_buffer[host_indices[j]], kv_buffer[j])
    

def test_batch_operation_mha(tp_size, page_first):
    """MHA test the batch set/get APIs with multiple key-value pairs"""
    print("=" * 100)

    keys, host_indices, mem_pool_host = generate_data_and_get_pool_host(is_mla=False)
    test_dir = "/home/ucm_storage_data_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for i in range(tp_size - 1, -1, -1):
        config = HiCacheStorageConfig(
            tp_rank=i,
            tp_size=tp_size,
            is_mla_model=False,
            is_page_first_layout=page_first,
            model_name="test",
            extra_config = {
                "kv_connector_extra_config": {
                    "ucm_connector_name": "UcmNfsStore",
                    "ucm_connector_config": {
                        "storage_backends": "/home/ucm_storage_data_test",
                        "max_cache_size": 10240,
                        "kv_block_size": 262144
                    }
                }
            }
        )

        store = UnifiedCacheStore(
            storage_config=config, mem_pool_host=mem_pool_host, tp_group=None)
        store.register_mem_pool_host(mem_pool_host)

        exist_result = store.batch_exists(keys)
        expected = 0
        assert exist_result == expected

        set_results = store.batch_set_v1(keys, host_indices)
        expected = [True] * len(set_results)
        assert set_results == expected

        if i == 0:
            exist_result = store.batch_exists(keys)
            expected = len(keys)
            assert exist_result == expected

            kv_buffer = mem_pool_host.kv_buffer
            for j in range(len(host_indices)):
                mem_pool_host.kv_buffer[:, host_indices[j]].zero_()
                assert torch.all(mem_pool_host.kv_buffer[:, host_indices[j]] == 0)

            get_results = store.batch_get_v1(keys, host_indices)
            expected = [True] * len(get_results)
            assert get_results == expected

            for j in range(len(host_indices)):
                torch.equal(mem_pool_host.kv_buffer[:, host_indices[j]], kv_buffer[:, j])
from amazon_er.infra import resources


def test_resource_monitor_without_gpu(monkeypatch):
    monkeypatch.setattr(resources, "query_gpu", lambda: {"available": False, "utilization": None, "used": None, "total": None})
    sample = resources.ResourceMonitor("unit", country="US", source="S2", shard="0").sample(processed_rows=10, total_rows=20)
    assert sample.rss_bytes > 0
    assert sample.available_ram_bytes > 0
    assert not sample.gpu_available
    assert sample.gpu_memory_used_bytes is None


def test_resource_monitor_gpu_values_when_available(monkeypatch):
    monkeypatch.setattr(resources, "query_gpu", lambda: {"available": True, "utilization": 50.0, "used": 1024, "total": 4096})
    sample = resources.ResourceMonitor("unit").sample()
    assert sample.gpu_available
    assert sample.gpu_utilization_percent == 50.0
    assert sample.gpu_memory_used_bytes == 1024

"""Inspect the installed HCCL options ABI without creating communicators."""

import json


def main():
    import torch_npu

    result = {"torch_npu": torch_npu.__version__, "communicator_created": False}
    try:
        cls = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options
        options = cls()
        result["options_fields"] = [
            name for name in dir(options) if not name.startswith("_")
        ]
        result["hccl_config_before"] = str(getattr(options, "hccl_config", None))
        options.hccl_config = {"hccl_buffer_size": 256}
        result["hccl_config_assignment"] = str(options.hccl_config)
        result["runtime_allocation_verified"] = False
    except (AttributeError, TypeError, RuntimeError) as exc:
        result["error"] = str(exc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

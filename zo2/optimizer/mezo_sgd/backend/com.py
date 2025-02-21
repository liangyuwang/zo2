import torch

def compute_module_impl(
        module_dual_forward,
        module: torch.nn.Module,
        optimize_method: str,
        *args, **kwargs
    ):
    match optimize_method:
        case "":
            pass
        case "torch.compile":   # may introduce some precision mismatch
            module = torch.compile(module, *args, **kwargs)
        case _:
            raise NotImplementedError
    return module_dual_forward(module=module, *args, **kwargs)

def compute_function_impl(
        function_dual_forward,
        fn,
        optimize_method: str,
        *args, **kwargs
    ):
    match optimize_method:
        case "":
            pass
        case "torch.jit.script":   # may introduce some precision mismatch
            fn = torch.jit.script(fn, *args, **kwargs)
        case _:
            raise NotImplementedError
    return function_dual_forward(fn, *args, **kwargs)
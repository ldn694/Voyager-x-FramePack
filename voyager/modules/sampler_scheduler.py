class SamplerScheduler:
    def __init__(self, num_steps, modes):
        self.num_steps = num_steps
        self.modes = modes if isinstance(modes, list) else [modes]

    def default_scheduler(self, args):
        if len(self.modes) > 2:
            raise ValueError("Only support up to 2 modes.")

        if len(self.modes) == 1:
            return [self.modes[0]] * self.num_steps

        ratio = getattr(args, "ratio", None)
        if ratio is None:
            raise ValueError("Please provide 'ratio' argument for mixed modes.")

        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        split_point = int(self.num_steps * ratio)
        return [self.modes[0]] * split_point + [self.modes[1]] * (self.num_steps - split_point)
    
    def alternate_scheduler(self, args):
        if len(self.modes) == 2:
            raise ValueError("Alternate scheduler requires exactly 2 modes.")
        result = self.default_scheduler(args)

        start = getattr(args, "start", 0)
        if start < 0 or start >= self.num_steps:
            raise ValueError(f"start must be in [0, {self.num_steps - 1}], got {start}")
        
        step_interval = getattr(args, "step_interval", 1)
        if step_interval <= 0:
            raise ValueError(f"step_interval must be positive, got {step_interval}")

        for i in range(start, self.num_steps, step_interval):
            result[i] = self.modes[1] 
        return result

    def get_scheduler(self, method_name="default_scheduler", args=None):
        try:
            method = getattr(self, method_name)
        except AttributeError:
            raise ValueError(f"Invalid method name: {method_name}")

        if not callable(method):
            raise ValueError(f"'{method_name}' exists but is not callable")

        return method(args)

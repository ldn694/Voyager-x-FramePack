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
        
        # =========================================================
        # DYNAMIC SANDWICH MODE & SPECIAL FLAGS
        # =========================================================
        if ratio > 1.0:
            
            # --- SPECIAL FALLBACK: ratio == 99 ---
            if ratio == 99.0 or ratio == 99:
                # First 10%: mode 0
                # Last 10%: mode 1
                start_steps = int(self.num_steps * 0.1)
                end_steps = int(self.num_steps * 0.1)
                mid_steps = self.num_steps - start_steps - end_steps
                
                # Middle 80%: Interleaved, mode 0 appears more (2:1 ratio)
                # Pattern: custom, custom, default
                pattern = [self.modes[0], self.modes[0], self.modes[1]]
                
                # Fill the middle block by repeating the pattern
                mid_schedule = [pattern[i % len(pattern)] for i in range(mid_steps)]
                
                return (
                    [self.modes[0]] * start_steps + 
                    mid_schedule + 
                    [self.modes[1]] * end_steps
                )

            # --- STANDARD SANDWICH (e.g. 10, 20, 30) ---
            if ratio >= 50.0:
                raise ValueError("Sandwich percentage must be less than 50 (unless ratio is 99).")

            # Convert 20 -> 0.2
            bread_ratio = ratio / 100.0 
            
            split_1 = int(self.num_steps * bread_ratio)
            split_2 = int(self.num_steps * (1.0 - bread_ratio))

            return (
                [self.modes[1]] * split_1 +
                [self.modes[0]] * (split_2 - split_1) + # mode0 is custom, mode1 is default
                [self.modes[1]] * (self.num_steps - split_2)
            )

        # =========================================================
        # NORMAL BEHAVIOR: Standard 2-Phase Mode (ratio 0.0 to 1.0)
        # =========================================================
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        split_point = int(self.num_steps * ratio)
        return [self.modes[0]] * split_point + [self.modes[1]] * (self.num_steps - split_point)
    
    def alternate_scheduler(self, args):
        result = self.default_scheduler(args)

        starts = getattr(args, "start", [0])
        for start in starts:
            if start < 0 or start >= self.num_steps:
                raise ValueError(f"start must be in [0, {self.num_steps - 1}), got {start}")
        
        ends = getattr(args, "end", [self.num_steps])
        for end in ends:
            if end < 0 or end > self.num_steps:
                raise ValueError(f"end must be in [0, {self.num_steps}], got {end}")

        step_intervals = getattr(args, "step_interval", [1])
        for step_interval in step_intervals:
            if step_interval <= 0:
                raise ValueError(f"step_interval must be positive, got {step_interval}")
        
        assert len(starts) == len(ends) == len(step_intervals), "start, end, and step_interval lists must have the same length"
        for start, end, step_interval in zip(starts, ends, step_intervals):
            result[start:end:step_interval] = [self.modes[1]] * len(result[start:end:step_interval])

        return result

    def get_scheduler(self, method_name="default_scheduler", args=None):
        try:
            method = getattr(self, method_name)
        except AttributeError:
            raise ValueError(f"Invalid method name: {method_name}")

        if not callable(method):
            raise ValueError(f"'{method_name}' exists but is not callable")

        return method(args)
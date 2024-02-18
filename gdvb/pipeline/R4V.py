class R4V:
    def __init__(self, options):
        self._executor = "$GDVB/scripts/run_R4V.sh"
        self._pre_params, self._post_params = self._parse_options(options)

    @staticmethod
    def _parse_options(options):
        pre_params = []
        post_params = []
        for op in options:
            if op == "distill":
                pre_params += ["distill"]
            elif op == "debug":
                post_params += ["--debug"]
            else:
                raise NotImplementedError(op)
        return pre_params, post_params

    def execute(self, params):
        cmd = self._executor
        for p in self._pre_params:
            cmd += f" {p}"
        for p in params:
            cmd += f" {p}"
        for p in self._post_params:
            cmd += f" {p}"
        return cmd
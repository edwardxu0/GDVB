class SwarmHost:
    def __init__(self, options):
        self._executor = "$GDVB/scripts/run_SwarmHost.sh"
        self._pre_params, self._post_params, self.verifier_name = self._parse_options(
            options
        )

    @staticmethod
    def _parse_options(options):
        pre_params = []
        post_params = []
        verifier_name = None
        for op in options:
            if op in [
                "abcrown",
            ]:
                verifier_name = op
                post_params += [f"--verifier {op}"]
            elif op == "debug":
                post_params += ["--debug"]
            else:
                raise NotImplementedError
        assert verifier_name is not None
        return pre_params, post_params, verifier_name

    def execute(self, params):
        cmd = self._executor
        for p in self._pre_params:
            cmd += f" {p}"
        for p in params:
            cmd += f" {p}"
        for p in self._post_params:
            cmd += f" {p}"
        return cmd

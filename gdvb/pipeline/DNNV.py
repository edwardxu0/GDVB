class DNNV:
    def __init__(self, options):
        self._executor = "./scripts/run_DNNV.sh"
        self._pre_params, self._post_params, self.verifier_name = self._parse_options(
            options
        )

    @staticmethod
    def _parse_options(options):
        pre_params = []
        post_params = []
        verifier_name = None
        for op in options:
            if "eran" in op:
                verifier_name = op
                post_params += [f'--eran --eran.domain {op.split("_")[1]}']
            elif op == "bab_sb":
                verifier_name = op
                post_params += ["--bab --bab.smart_branching True"]
            elif op in [
                "neurify",
                "planet",
                "bab",
                "reluplex",
                "dnnf",
                "marabou",
                "nnenum",
                "mipverify",
                "verinet",
            ]:
                verifier_name = op
                post_params += [f"--{op}"]
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


class DNNV_wb(DNNV):
    def __init__(self, options):
        super().__init__(options)
        self._executor = "./scripts/run_DNNV_wb.sh"

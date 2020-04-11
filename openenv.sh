source .venv/bin/activate

# GDVB
export GDVB=`pwd`
export PYTHONPATH="${PYTHONPATH}:${GDVB}"

# r4v
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/r4v/"

# dnnv
export DNNV_DIR="${GDVB}/lib/dnnv"
export PATH=$DNNV_DIR/bin/:$PATH
export LD_LIBRARY_PATH=$DNNV_DIR/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$DNNV_DIR:$PYTHONPATH
    # gurobi paths
export GUROBI_HOME=$DNNV_DIR/bin/gurobi810/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
    # eran paths
export PYTHONPATH=$DNNV_DIR/lib/eran/tf_verify:$PYTHONPATH
export PYTHONPATH=$DNNV_DIR/lib/eran/ELINA/python_interface:$PYTHONPATH
    # julia paths
export PATH=$DNNV_DIR/bin/julia-1.0.4/bin:$PATH
    # bab paths
export PYTHONPATH=$DNNV_DIR/lib/PLNN-verification/convex_adversarial:$PYTHONPATH

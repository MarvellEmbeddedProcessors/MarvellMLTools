Marvell(R) supports a family of high performance Data Processing
Units (DPUs) with integrated compute, high speed I/O and workload
accelerators. These workload accelerators include Marvell's
Machine Learning Inference Processor (MLIP), a highly optimized,
integrated inference engine.
This repo hosts Marvell's ML toolchain to deploy ML models on the MLIP:
1. /bin/mrvl-tmlc is an ML compiler
2. /bin/mrvl-mlsim is a software simulator tool to simulate the MLIP hardware
This toolchain is integrated with the Apache TVM project: https://github.com/apache/tvm
After cloning the repo, adding the /bin folder to the $PATH will enable the TVM compiler & runtime to access the toolchain:
export PATH=$PATH:/path_to_repo/bin
For a detailed usage guide, please refer to: https://github.com/apache/tvm/blob/main/docs/how_to/deploy/mrvl.rst
NOTE: The toolchain is tested on an x86 platform running Ubuntu 22.04

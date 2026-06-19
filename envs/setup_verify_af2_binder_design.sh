conda activate af2_binder_design

# Symlink PyRosetta from dl_binder_design
DEST="~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib/python3.11/site-packages"
SRC="~/Desktop/pmhc_binder_kras/envs/dl_binder_design/lib/python3.11/site-packages"
ln -s ${SRC}/pyrosetta ${DEST}/pyrosetta
ln -s ${SRC}/rosetta ${DEST}/rosetta
ln -s "${SRC}/pyrosetta-2026.3+releasequarterly.5e498f1409.dist-info" \
      "${DEST}/pyrosetta-2026.3+releasequarterly.5e498f1409.dist-info"

# Create cusolver symlinks
CUSOLVER_DIR=${DEST}/nvidia/cusolver/lib
ln -sf ${CUSOLVER_DIR}/libcusolver.so.11 ${CUSOLVER_DIR}/libcusolver.so
ln -sf ${CUSOLVER_DIR}/libcusolverMg.so.11 ${CUSOLVER_DIR}/libcusolverMg.so

# Create cuDNN 8.9 symlink
CUDNN_DIR=${DEST}/nvidia/cudnn/lib
ln -sf ${CUDNN_DIR}/libcudnn.so.8 ${CUDNN_DIR}/libcudnn.so

# Verify
python -c "
import jax, tensorflow as tf, ml_dtypes, Bio
print('JAX:        ', jax.__version__)
print('TensorFlow: ', tf.__version__)
print('ml_dtypes:  ', ml_dtypes.__version__)
print('Biopython:  ', Bio.__version__)
print('float4_e2m1fn:', hasattr(ml_dtypes, 'float4_e2m1fn'))
print('JAX devices:', jax.devices())
from Bio.Data import SCOPData
print('SCOPData: OK')
from pyrosetta import *
from rosetta import *
print('PyRosetta: OK')
"
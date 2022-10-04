from hypertiling.static.static_rotational_improved import KernelStaticRotationalImproved

# for njiting the functions
import sys
import io

__std_out = sys.stdout
sys.stdout = io.StringIO()

tiling = KernelStaticRotationalImproved(3, 7, 2, "cell")

sys.stdout = __std_out
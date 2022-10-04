from hypertiling.generative.generative_reflection import KernelGenerativeReflection

# for njiting the functions
import sys
import io

__std_out = sys.stdout
sys.stdout = io.StringIO()

tiling = KernelGenerativeReflection(3, 7, 2)

sys.stdout = __std_out
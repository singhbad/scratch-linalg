import time
import numpy as np

def pure_python_dot(u, v):
    return sum(a * b for a, b in zip(u, v))

n = 1_000_000
u_list = [1.0] * n
v_list = [2.0] * n
u_np = np.ones(n)
v_np = np.ones(n) * 2

# Pure Python timing
start = time.perf_counter()
result_py = pure_python_dot(u_list, v_list)
py_time = time.perf_counter() - start

# NumPy timing
start = time.perf_counter()
result_np = u_np.dot(v_np)
np_time = time.perf_counter() - start

print(f"Vector length: {n:,}")
print(f"Pure Python dot product: {py_time*1000:.3f} ms  (result={result_py})")
print(f"NumPy dot product:       {np_time*1000:.3f} ms  (result={result_np})")
print(f"Speedup: {py_time/np_time:.1f}x")

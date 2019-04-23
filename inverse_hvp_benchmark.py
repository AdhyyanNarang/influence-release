import numpy as np
from scipy.optimize import fmin_ncg
import time

np.random.seed(32)

dim = 10000
lamb = 1
A = np.random.rand(dim,dim)

H = A.T.dot(A) + lamb * np.eye(dim)
v = np.random.rand(dim)

# print("SOLVING BY INVERTING")
# start = time.time()
# direct_inv = np.linalg.inv(H).dot(v)
# end = time.time()
# print("TIME:", end - start)
# print("")


print("SOLVING IMPLICITLY")
start = time.time()
solver_inv = np.linalg.solve(H, v)
end = time.time()
print("TIME:", end - start)
print("")


loss = lambda x: 0.5 * np.dot(H.dot(x), x) - np.dot(v, x)
grad = lambda x: H.dot(x) - v
hess = lambda x: H
hess_prod = lambda x, p: H.dot(p)

print("SOLVING USING NCG WITH IMPLICIT HESSIAN")
start = time.time()
ncg_hess_implicit_inv = fmin_ncg(f=loss,
                                 x0=v,
                                 fprime=grad,
                                 fhess_p=hess_prod,
                                 avextol=1e-8,
                                 maxiter=100)
end = time.time()
print("TIME:", end - start)
print("")

print("SOLVING USING NCG WITH EXPLICIT HESSIAN")
start = time.time()
ncg_hess_explicit_inv = fmin_ncg(f=loss,
                                 x0=v,
                                 fprime=grad,
                                 fhess=hess,
                                 avextol=1e-8,
                                 maxiter=100)
end = time.time()
print("TIME:", end - start)
print("")


__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-08-20"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from sandia_vel import mesh, parameters, array, linspace, log, minimum, maximum, R1, R2, H, L, re_high, omega, nu, max_error, velocity_degree, max_iters, case, cases, jet, pilot, coflow, outlet, centerline, border, Pilot, jete, pilote, coflowe, V, Q, DG, VQ, up0_ 

omega = 0.01

VQ = MixedFunctionSpace([V, Q, Q, Q, Q])

up = TrialFunction(VQ)
vq = TestFunction(VQ)
v, q, v_k, v_e, v_nut = split(vq)
u, p, k, e, nut = split(up)

up_ = Function(VQ)
u_, p_, k_, e_, nut_ = split(up_)

assign(up_.sub(0), up0_.sub(0))
assign(up_.sub(1), up0_.sub(1))

turb_dofs = [VQ.sub(2).dofmap().dofs(),
             VQ.sub(3).dofmap().dofs(),
             VQ.sub(4).dofmap().dofs()]

# Initialize
up_.vector()[turb_dofs[0]] = 1
up_.vector()[turb_dofs[1]] = 1e-3
up_.vector()[turb_dofs[2]] = 0.09*1e3

# Get turbulence dofs
ll = []
for i in range(len(turb_dofs)):
    ll += list(turb_dofs[i]) 
ll = array(ll)

up_1 = Function(VQ)
up_1.vector()[:] = up_.vector()

model_prm = dict(
    Cmu = Constant(0.09),
    Ce1 = Constant(1.44),
    Ce2 = Constant(1.92),
    sigma_e = Constant(1.3),
    sigma_k = Constant(1.0))
vars().update(model_prm)

r = Expression("x[1]", domain=mesh)
n = FacetNormal(mesh)
#nut_ = Constant(100.0)

F_nut = (nut_ * v_nut - Cmu * k_ * k_ / e_ * v_nut) * r * dx()
P_ = 2*inner(grad(u_), sym(grad(u_)))*nut_

Fk = (nu + nut_)*inner(grad(v_k), grad(k_))*r*dx() \
      + inner(v_k, dot(grad(k_), u_))*r*dx() \
      - P_*v_k*r*dx() + e_*v_k*r*dx()
        
Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e_))*r*dx() \
        + inner(v_e, dot(grad(e_), u_))*r*dx() \
        - (Ce1*P_ - Ce2*e_)*(e_/k_)*v_e*r*dx()
        
Ft = Fk + Fe + F_nut

F_NS = inner(dot(grad(u_), u_), v)*r*dx() \
       + (nu+nut_)*inner(grad(u_), grad(v))*r*dx() + (nu+nut_)*u_[1]*v[1]/r*dx() \
       - inner(p_, (r*v [1]).dx(1) + r*v [0].dx(0))*dx() \
       - inner(q , (r*u_[1]).dx(1) + r*u_[0].dx(0))*dx()

Ft += F_NS

bc0 = DirichletBC(VQ.sub(0),   jete,  jet)
bc1 = DirichletBC(VQ.sub(0), pilote,  pilot)
bc2 = DirichletBC(VQ.sub(0), coflowe,  coflow)
bc3 = DirichletBC(VQ.sub(0).sub(1), 0, centerline)
bc4 = DirichletBC(VQ.sub(2), 1e-3,  jet)
bc5 = DirichletBC(VQ.sub(2), 1e-4,  pilot)
bc6 = DirichletBC(VQ.sub(2), 1e-5,  coflow)
bc7 = DirichletBC(VQ.sub(3), 1e-3,  jet)
bc8 = DirichletBC(VQ.sub(3), 1e-4,  pilot)
bc9 = DirichletBC(VQ.sub(3), 1e-5,  coflow)

bcs = [bc0, bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8, bc9]
#for bc in bcs:
    #bc.apply(up_ .vector())
    #bc.apply(up_1.vector())

J = derivative(Ft, up_, up)
A = Matrix()

up_sol = LUSolver("mumps")
up_sol.parameters["same_nonzero_pattern"] = True

def iterate(max_iters, omega, A, b, max_error, up_, up_1, bcs, J, Ft, **kwargs):
    # Newton iterations for steady flow
    iter = 0
    error = 1
    while iter < max_iters and error > max_error: 
        A = assemble(J, tensor=A)
        for bc in bcs:
            bc.apply(A)

        up_1.vector().zero()
        up_sol.solve(A, up_1.vector(), b)
        up_.vector().axpy(-omega, up_1.vector())
        
        # Enforce lower and upper boundary
        xa = up_.vector().array()
        xa[ll] = minimum(maximum(1e-10, xa[ll]), 1e4)
        up_.vector().set_local(xa)
        
        b = assemble(Ft, tensor=b)
        for bc in bcs:
            bc.apply(b, up_.vector())

        # Update to next iteration
        up_1.vector().zero(); up_1.vector().axpy(1.0, up_.vector())
            
        error = b.norm('l2') 
        
        print iter, " ", error

        iter += 1

# Assemble rhs once, before entering iterations (velocity components)
b = assemble(Ft)
for bc in bcs:
    bc.apply(b, up_.vector())

max_iters = 10
iterate(**vars())

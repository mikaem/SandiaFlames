__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-08-20"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from numpy import array, linspace, log, minimum, maximum

import os

parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["quadrature_degree"] = 2 
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 --fast-math"

if not os.path.isfile("../mesh/Sandia.xml"):
    try:
        os.system("gmsh ../mesh/Sandia.geo -2 -o ../mesh/Sandia.msh")
        os.system("dolfin-convert ../mesh/Sandia.msh ../mesh/Sandia.xml")
        os.system("rm ../mesh/Sandia.msh")
    except RuntimeError:
        raise "Gmsh is required to run this program"
      
mesh = Mesh("../mesh/Sandia.xml")

R1 = 3.6e-3 # m
R2 = 9.1e-3 
H = 0.15
L = 0.3
Lm = 2*0.07*R1
re_high = False
omega = 0.1
nu = 1.58e-5  # m**2/s
max_error = 1e-8
velocity_degree = 2
max_iters = 25
Cmu = 0.09

##
case = "D"
##
cases = {
      "C": {'U_jet': 29.7, # m/s
            'U_pilot': 6.8,
            'U_coflow': 0.9},
      
      "D": {'U_jet': 49.6,
            'U_pilot': 11.4,
            'U_coflow': 0.9}
      }

# inlet = 1
def jet(x, on_boundary): 
    return on_boundary and  x[1] < R1 + DOLFIN_EPS_LARGE and x[0] < DOLFIN_EPS_LARGE
  
def pilot(x, on_boundary): 
    return on_boundary and  x[1] > R1 - DOLFIN_EPS_LARGE and x[1] < R2 + DOLFIN_EPS_LARGE and x[0] < DOLFIN_EPS_LARGE

def coflow(x, on_boundary): 
    return on_boundary and  x[1] > R2 - DOLFIN_EPS_LARGE and x[0] < DOLFIN_EPS_LARGE
      
# outlet = 2
def outlet(x, on_boundary): 
    return on_boundary and x[0] > L - DOLFIN_EPS_LARGE or x[1] > H - DOLFIN_EPS_LARGE

def centerline(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS_LARGE

def border(x, on_boundary):
    return on_boundary and x[1] > H - 10*DOLFIN_EPS_LARGE

class KEJet(Expression):
    def __init__(self):
        # Coefficients from Ahmad
        # uu fit coefs
        self.p1uu = (6.136e8, -9.219e8, 2.149e9)
        self.p2uu = (-6.806e5, -8.87e6, 7.509e6)
        self.p3uu = (3681.0, -8111.0, 1.547e4)
        self.p4uu = (5.467, 1.044, 9.89)

        # vv fit coefs
        self.p1vv = (-2.609e8, -3.687e8, -1.531e8)
        self.p2vv = (1.204e6, 6.287e5, 1.778e6)
        self.p3vv = (-481.0, -1309.0, 346.7)
        self.p4vv = (2.207, 1.896, 2.517)
  
    def eval(self, values, x):
        uu = 0; vv = 0
        if x[1] <= R1:
            if case == "D":
                uu = self.p1uu[0]*pow(x[1],3.0) + self.p2uu[0]*pow(x[1],2.0) + self.p3uu[0]*x[1] + self.p4uu[0]
                vv = self.p1vv[0]*pow(x[1],3.0) + self.p2vv[0]*pow(x[1],2.0) + self.p3vv[0]*x[1] + self.p4vv[0]
            elif case == "E":
                uu = self.p1uu[1]*pow(x[1],3.0) + self.p2uu[1]*pow(x[1],2.0) + self.p3uu[1]*x[1] + self.p4uu[1]
                vv = self.p1vv[1]*pow(x[1],3.0) + self.p2vv[1]*pow(x[1],2.0) + self.p3vv[1]*x[1] + self.p4vv[1]            
            elif case == "F":
                uu = self.p1uu[2]*pow(x[1],3.0) + self.p2uu[2]*pow(x[1],2.0) + self.p3uu[2]*x[1] + self.p4uu[2]
                vv = self.p1vv[2]*pow(x[1],3.0) + self.p2vv[2]*pow(x[1],2.0) + self.p3vv[2]*x[1] + self.p4vv[2]            
        values[0] = max(0.5*(uu + 2*vv), 0)
        values[1] = pow(Cmu, 0.75)*pow(values[0], 1.5) / Lm
        
    def value_shape(self):
        return (2,)

class LaminarPilot(Expression):
    
    def __init__(self):
        self.C = C = (R1**2-R2**2)/log(R2/R1)
        self.D = D = -C*log(R1)-R1**2
        self.A = cases[case]['U_pilot'] / (0.25*(R2**4-R1**4) + 
                                           0.25*C*(R2**2*(2*log(R2)-1) - R1**2*(2*log(R1)-1))
                                           +0.5*D*(R2**2-R1**2)) * (0.5*(R2**2-R1**2))
        
    def eval(self, values, x):
        values[0] = self.A*(x[1]*x[1]+self.C*log(x[1])+self.D)
        values[1] = 0.0        
        
    def value_shape(self):
        return (2,)
    
laminarjet = Expression(("U * (r*r-x[1]*x[1])", "0"), 
                         U=2*cases[case]['U_jet']/R1**2, r=R1)

laminarpilot = LaminarPilot()
laminarcoflow = Expression(("U", "0"), U=cases[case]['U_coflow'])

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
DG = FunctionSpace(mesh, "DG", 0)
VQ = MixedFunctionSpace([V, Q])
KE = MixedFunctionSpace([Q, Q])
r = Expression("x[1]", domain=mesh)
n = FacetNormal(mesh)

up = TrialFunction(VQ)
vq = TestFunction(VQ)
v, q = split(vq)
u, p = split(up)

# Turbulence
nut_ = Function(DG)
ke = TrialFunction(KE)
kev = TestFunction(KE)
v_k, v_e = split(kev)
k, e = split(ke)
tt = TrialFunction(DG) 
tg = TestFunction(DG)

ke_ = Function(KE)
dke = Function(KE)
k_, e_ = split(ke_)

# Create effective viscosity - the sum of viscosity and turbulent viscosity
nut = nu + nut_
up_ = Function(VQ)
u_, p_ = split(up_)
dup = Function(VQ)

bc0 = DirichletBC(VQ.sub(0), laminarjet,  jet)
bc1 = DirichletBC(VQ.sub(0), laminarpilot,  pilot)
bc2 = DirichletBC(VQ.sub(0), laminarcoflow,  coflow)
bc3 = DirichletBC(VQ.sub(0).sub(1), 0, centerline)
bcs = [bc0, bc1, bc2, bc3]

# Variational form of Navier-Stokes in cylinder coordinates
NS = inner(dot(grad(u_), u_), v)*r*dx() \
     + 2*nut*inner(sym(grad(u_)), grad(v))*r*dx() + nut*u_[1]*v[1]/r*dx() \
     - inner(p_, (r*v [1]).dx(1) + r*v [0].dx(0))*dx() \
     - inner(q , (r*u_[1]).dx(1) + r*u_[0].dx(0))*dx() \
     - nut*inner(dot(grad(u_).T, n), v)*ds()

J = derivative(NS, up_, up)
up_sol = LUSolver("mumps")
up_sol.parameters["same_nonzero_pattern"] = True
A = Matrix()
b = Vector(up_.vector())

# k-epsilon model
model_prm = dict(
    Cmu = Constant(0.09),
    Ce1 = Constant(1.44),
    Ce2 = Constant(1.92),
    sigma_e = Constant(1.3),
    sigma_k = Constant(1.0))
vars().update(model_prm)

P_ = 2*inner(grad(u_), sym(grad(u_)))*nut_

Fk = nut*inner(grad(v_k), grad(k_))*r*dx() \
   + inner(v_k, dot(grad(k_), u_))*r*dx() \
   - P_*v_k*r*dx() + e_*v_k*r*dx()
        
Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e_))*r*dx() \
        + inner(v_e, dot(grad(e_), u_))*r*dx() \
        - (Ce1*P_ - Ce2*e_)*(e_/k_)*v_e*r*dx()
        
Fke = Fk + Fe
Jke = derivative(Fke, ke_, ke)
Ake = Matrix()

F_nut = (tt*tg - Cmu * k_ * k_ / e_ * tg) * r * dx()
A_nut = assemble(lhs(F_nut))

bc4 = DirichletBC(KE, KEJet(),  jet)
bc5 = DirichletBC(KE, (1., 10.),  pilot)
bc6 = DirichletBC(KE, (0.1, 1.),  coflow)
bcs_ke = [bc4, bc5, bc6]

nut_sol = LUSolver("mumps")
nut_sol.parameters["reuse_factorization"] = True
x = Vector(nut_.vector())

ke_sol = LUSolver("mumps")
ke_sol.parameters["same_nonzero_pattern"] = True
omega_nut = 0.1
omega_ke = 0.1

turb_dofs = [KE.sub(0).dofmap().dofs(),
             KE.sub(1).dofmap().dofs()]

def solve_nut(omega_nut):
    b_nut = assemble(rhs(F_nut))
    nut_sol.solve(A_nut, x, b_nut)
    x.axpy(-1, nut_.vector())
    nut_.vector().axpy(omega_nut, x)
    nut_.vector().set_local(minimum(maximum(1e-10, nut_.vector().array()), 1.0))

def velocity_iter(A, b, omega):
    # Newton iterations for steady flow
    A = assemble(J, tensor=A)
    for bc in bcs:
        bc.apply(A)

    dup.vector().zero()
    up_sol.solve(A, dup.vector(), b)
    up_.vector().axpy(-omega, dup.vector())
    
    b = assemble(NS, tensor=b)
    for bc in bcs:
        bc.apply(b, up_.vector())

    return b.norm('l2') 

def ke_iter(Ake, bke, omega_ke):
    # Newton iterations for steady flow
    error = 1
    Ake = assemble(Jke, tensor=Ake)
    for bc in bcs_ke:
        bc.apply(Ake)

    dke.vector().zero()
    ke_sol.solve(Ake, dke.vector(), bke)
    ke_.vector().axpy(-omega_ke, dke.vector())
    
    # Enforce lower and upper boundary
    xa = ke_.vector().array()
    ke_.vector().set_local(minimum(maximum(1e-10, xa), 1e6))
    
    bke = assemble(Fke, tensor=bke)
    for bc in bcs_ke:
        bc.apply(bke, ke_.vector())

    return bke.norm('l2') 
    
# init u
# Assemble rhs once, before entering iterations (velocity components)
# initialize nut_
nut_.vector()[:] = 0.1

for bc in bcs:
    bc.apply(up_.vector())
b = assemble(NS)
for bc in bcs:
    bc.apply(b, up_.vector())
error = velocity_iter(A, b, 1.0)

# init ke
k0 = project(0.05*dot(u_, u_)*r, Q)
ke_.vector()[turb_dofs[0]] = k0.vector()[:]
ke_.vector().set_local(minimum(maximum(1e-10, ke_.vector().array()), 1e6))
e0 = project(pow(Cmu, 0.75)*pow(k_, 1.5)/Lm*r, Q)
ke_.vector()[turb_dofs[1]] = e0.vector()[:]
ke_.vector().set_local(minimum(maximum(1e-10, ke_.vector().array()), 1e6))
#solve_nut(1.0)

for bc in bcs_ke:
    bc.apply(ke_.vector())    
bke = assemble(Fke)
for bc in bcs_ke:
    bc.apply(bke, ke_.vector())

max_iters = 10 
iter = 0
error = 1
while iter < 10 and error > 1e-6: 
    error = velocity_iter(A, b, omega)
    error_ke = ke_iter(Ake, bke, omega_ke)
    solve_nut(omega_nut)
    iter += 1
    print iter, " ", error, error_ke

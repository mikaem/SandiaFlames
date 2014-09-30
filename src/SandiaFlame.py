__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-08-20"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from numpy import array, linspace, log, minimum, maximum
from collections import defaultdict
import numpy
import json
import os, sys

parameters["linear_algebra_backend"] = "PETSc"
#parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
#parameters["form_compiler"]["quadrature_degree"] = 2 
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 --fast-math"
set_log_active(False)

params = dict(
    R1 = 3.6e-3,  # Inner radius pilot
    R2 = 9.1e-3,  # Outer radius pilot
    H = 0.15,     # Length of mesh
    L = 0.4,      # Height of mesh
    cl4 = 0.001, # Mesh density (L, 0)
    cl3 = 0.005,  # Mesh density (L, H)
    cl2 = 0.005,  # Mesh density (0, H)
    cl1 = 0.0001, # Mesh density (0, 0)
    case = "D",
    model = "CR", #Velocity-pressure function spaces
    beta = 0.01,  # Numerical stability parameter
    runDG = False,# DG model or not (experimental)
    nu = Constant(1.58e-5),
    max_error = 1e-7,
    coupled = False,
    max_iters = 200,
    sigma = 0.00075 # Parameter used to create smooth inlet profiles. Lower value -> sharper Heaviside = less stable
)

# Any parameter may be overloaded on command line
commandline_kwargs = {}
for s in sys.argv[1:]:
    if s.count('=') == 1:
        key, value = s.split('=', 1)
    else:
        raise TypeError(s+" Only kwargs separated with '=' sign allowed")
    try:
        value = json.loads(value) 
    except ValueError:
        if value in ("True", "False"): # json understands true/false, but not True/False
            value = eval(value)
    if key in params:
        params[key] = value
    else:
        commandline_kwargs[key] = value

# Create a Gmsh-mesh and load it
meshcode = """
Point(1) = {0, 0, 0, %(cl1)s};
Point(2) = {0, %(R1)s, 0, %(cl1)s};
Point(3) = {0, %(R2)s, 0, %(cl1)s};
Point(4) = {0, %(H)s, 0, %(cl2)s};
Point(5) = {%(L)s, %(H)s, 0, %(cl3)s};
Point(6) = {%(L)s, 0, 0, %(cl4)s};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(8) = {3, 4, 5, 6, 1, 2};
Plane Surface(8) = {8};
""" % params

if not os.path.isfile("../mesh/Sandia.xml") or ("remesh" in commandline_kwargs 
                                                and commandline_kwargs["remesh"] == True):
    f = open("../mesh/Sandia.geo", "w")
    f.write(meshcode)
    f.close()
    os.system("gmsh ../mesh/Sandia.geo -2 -o ../mesh/Sandia.msh")
    os.system("dolfin-convert ../mesh/Sandia.msh ../mesh/Sandia.xml")
    os.system("rm ../mesh/Sandia.msh")

mesh = Mesh("../mesh/Sandia.xml")

vars().update(params)
vars().update(commandline_kwargs)

Lm = 2*0.07*R1

# Underrelaxation factors
omega = defaultdict(lambda : 0.8, {"nut": 0.8})

##
cases = {
      "C": {'U_jet': 29.7, # m/s
            'U_pilot': 6.8,
            'U_coflow': 0.9},
      
      "D": {'U_jet': 49.6,
            'U_pilot': 11.4,
            'U_coflow': 0.9}
      }

# k-epsilon model coefficients
model_prm = dict(
    Cmu = Constant(0.09),
    Ce1 = Constant(1.6), # modification from 1.44
    Ce2 = Constant(1.92),
    sigma_e = Constant(1.3),
    sigma_k = Constant(1.0),
    e_d = Constant(0.0))

vars().update(model_prm)      

k_inlet = {"k_jet": 0.5*pow(cases[case]["U_jet"], 2)*0.01,
           "k_pilot": 0.5*pow(cases[case]["U_pilot"], 2)*0.01,
           "k_coflow": 0.5*pow(cases[case]["U_coflow"], 2)*0.005}

e_inlet = {"e_jet": pow(Cmu(0), 0.75)*pow(k_inlet["k_jet"], 1.5) / 0.001,
           "e_pilot": pow(Cmu(0), 0.75)*pow(k_inlet["k_pilot"], 1.5) / 0.0025,
           "e_coflow": pow(Cmu(0), 0.75)*pow(k_inlet["k_coflow"], 1.5) / 0.05}

smoothinlet = "0.5*(1.0+erf((R1-x[1])/sigma))*jet + cof + (1.0-cof)*0.5*(1+erf((R2-x[1])/sigma))*(pilot-cof)"

uin = Expression((smoothinlet, "0"), R1=R1, R2=R2, pilot=cases[case]["U_pilot"],
                 cof=cases[case]["U_coflow"], jet=cases[case]["U_jet"], sigma=sigma)

uin0 = Expression(smoothinlet, R1=R1, R2=R2, pilot=cases[case]["U_pilot"],
                 cof=cases[case]["U_coflow"], jet=cases[case]["U_jet"], sigma=sigma)

kin = Expression(smoothinlet, R1=R1, R2=R2, pilot=k_inlet["k_pilot"],
                 cof=k_inlet["k_coflow"], jet=k_inlet["k_jet"], sigma=sigma)

ein = Expression(smoothinlet, R1=R1, R2=R2, pilot=e_inlet["e_pilot"], 
                 cof=e_inlet["e_coflow"], jet=e_inlet["e_jet"], sigma=sigma)


def jet(x, on_boundary): 
    return on_boundary and  x[1] < R1 + 10*DOLFIN_EPS_LARGE and x[0] < 10*DOLFIN_EPS_LARGE
  
def pilot(x, on_boundary): 
    return on_boundary and  x[1] > R1 - 10*DOLFIN_EPS_LARGE and x[1] < R2 + 10*DOLFIN_EPS_LARGE and x[0] < 10*DOLFIN_EPS_LARGE

def coflow(x, on_boundary): 
    return on_boundary and  x[1] > R2 - 10*DOLFIN_EPS_LARGE and x[0] < 10*DOLFIN_EPS_LARGE
      
def inlet(x, on_boundary): 
    return on_boundary and x[0] < 10*DOLFIN_EPS_LARGE

def outlet(x, on_boundary): 
    return on_boundary and x[0] > L - 10*DOLFIN_EPS_LARGE or x[1] > H - 10*DOLFIN_EPS_LARGE

def centerline(x, on_boundary):
    return on_boundary and x[1] < 10*DOLFIN_EPS_LARGE

def border(x, on_boundary):
    return on_boundary and x[1] > H - 10*DOLFIN_EPS_LARGE

k_min = k_inlet["k_coflow"]
e_min = e_inlet["e_coflow"]

# Set lower and upper limits for computations
limits = {"k": (k_min/100, 500.),
          "e": (e_min/100, 1e7),
          "nut": (1e-6, 0.01),
          "T": (1e-6, 1e3),
          "ke": (k_min/100, 1e7)}

# Define relevant function spaces
Q = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)
DG1 = FunctionSpace(mesh, "DG", 1)
if model == "TH":
    V = VectorFunctionSpace(mesh, "CG", 2)
    S = Q
elif model == "CR":
    V = VectorFunctionSpace(mesh, "CR", 1)
    S = DG0
    
# Mixed function space for velocity and pressure
VQ = MixedFunctionSpace([V, S])

# Cylindrical coordinate
r = Expression("x[1]", domain=mesh)
r_inv = Expression("x[1] > 1e-8 ? 1.0/x[1] : 0.0", domain=mesh)

n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2

up = TrialFunction(VQ)
vq = TestFunction(VQ)
v, q = split(vq)
u, p = split(up)

up_ = Function(VQ, name="up")
u_, p_ = split(up_)

# Turbulence
if not coupled:
    # Space for k and epsilon
    TSpace = DG1 if runDG else Q
    k = TrialFunction(TSpace)
    v_k = TestFunction(TSpace)
    e = TrialFunction(TSpace)
    v_e = TestFunction(TSpace)

    k_ = Function(TSpace, name="k")
    e_ = Function(TSpace, name="e")

else:
    # Space for k and epsilon
    TSpace = Q*Q
    k, e = TrialFunctions(TSpace)
    v_k, v_e = TestFunctions(TSpace)
    ke_ = Function(TSpace, name="ke")
    k_, e_ = split(ke_)
    
# Create some dictionaries to hold work matrices and functions
class Mat_cache_dict(dict):
    def __missing__(self, key):
        A, sol = (assemble(key), LUSolver("mumps"))
        sol.parameters["reuse_factorization"] = True
        self[key] = (A, sol)
        return self[key]

class Fun_cache_dict(dict):
    def __missing__(self, key):
        self[key] = Function(key)
        return self[key]

A_cache = Mat_cache_dict()
Fun_cache = Fun_cache_dict()

def bound(x, low_lim=1e-8, upp_lim=1e8):
    x.set_local(minimum(maximum(low_lim, x.array()), upp_lim))
    x.apply("insert")

class derived_quantity(Function):
    """Derived Function with update method for underrelaxation
    """
    def __init__(self, name, Space, form, cyl=True, om=-1):
        Function.__init__(self, Space, name=name)
        self.form = form
        ta, tg = TrialFunction(Space), TestFunction(Space)
        if cyl: # update using r
            Mass = ta*tg*r*dx()
            bf = form*r*tg*dx()
        else:
            Mass = ta*tg*dx()
            bf = form*tg*dx()
        self.A, self.sol = A_cache[Mass]
        self.bf = bf
        self.b = Vector()
        self.work = Function(Space)
        self.om = omega[name] if om==-1 else om
        
    def update(self):
        assemble(self.bf, tensor=self.b)
        self.work.vector()[:] = self.vector()[:]
        self.sol.solve(self.A, self.vector(), self.b)        
        self.vector()._scale(self.om)
        self.vector().axpy(1-self.om, self.work.vector())
        bound(self.vector(), low_lim=limits[self.name()][0], upp_lim=limits[self.name()][1])
                    
# Define some derived quantities
# These may be used either as Function or as form. 
# Using it as a Function one may apply underrelaxation
T_ = derived_quantity("T", DG0, k_*(1/e_), cyl=False)    
T = T_.form

nut_ = derived_quantity("nut", DG0, Cmu * k_ * k_ / e_, cyl=False)
nut = nut_.form

P_ = Pr_ = derived_quantity("P", DG0, 2*nut_*inner(sym(grad(u_)), sym(grad(u_)))*r+2*nut_*u_[1]*u_[1]*r_inv, cyl=False)
Pr = Pr_.form

# Create effective viscosity - the sum of viscosity and turbulent viscosity
# A minimum of CellSize is added for stability. This mimicks the effect of upwinding
nutt = nu + nut_ + CellSize(mesh)*Constant(beta)
nute = nu + nut_*(1./sigma_e) + CellSize(mesh)*Constant(beta)

#vv = v + h*dot(grad(v), u_)
vv = v

#bc0 = DirichletBC(VQ.sub(0), one_over_seven_jet,  jet)
#bc1 = DirichletBC(VQ.sub(0), constantpilot,  pilot)
#bc2 = DirichletBC(VQ.sub(0), constantcoflow,  coflow)
bcin = DirichletBC(VQ.sub(0), uin, inlet)
bc3 = DirichletBC(VQ.sub(0).sub(1), 0, centerline)
bc4 = DirichletBC(VQ.sub(0).sub(1), 0, border)
#bcs = [bc0, bc1, bc2, bc3]

bcs = {}
bcs['up'] = [bcin, bc3, bc4]

# Variational form of Navier-Stokes in cylinder coordinates
U = 0.5*(u+u_)
NS =(inner(dot(grad(u_), u_), vv)*r*dx()
     + 2*nutt*inner(sym(grad(u_)), grad(vv))*r*dx() 
     + 2*nutt*u_[1]*vv[1]/r*dx()
     - inner(p_, (r*div(v) +v [1]))*dx()
     - inner(q , (r*div(u_)+u_[1]))*dx()
     - nutt*inner(dot(grad(u_).T, n), vv)*r*ds() )

# Picard
NS_P =(inner(dot(grad(u), u_), vv)*r*dx()
     + 2*nutt*inner(sym(grad(u)), grad(vv))*r*dx() 
     + 2*nutt*u[1]*vv[1]/r*dx()
     - inner(p, (r*div(v)+v[1]))*dx()
     - inner(q, (r*div(u)+u[1]))*dx()
     - nutt*inner(dot(grad(u).T, n), vv)*r*ds() 
     + inner(Constant((0, 0)), vv)*dx() )

J = derivative(NS, up_, up)
up_sol = LUSolver("mumps")
up_sol.parameters["same_nonzero_pattern"] = True

#vvk = mesh.hmin()*dot(grad(v_k), u_)
#vve = mesh.hmin()*dot(grad(v_e), u_)
vvk = v_k
vve = v_e

un = (dot(u_, n) + abs(dot(u_, n)))/2.0
# Penalty term
alpha = Constant(5000.0)

if runDG:
    Fke = { "k": dot(grad(v_k), nutt*grad(k) - u_*k)*r*dx \
            +avg(nutt)*(alpha/h('+'))*dot(jump(v_k, n), jump(k, n))*avg(r)*dS \
            -avg(nutt)*dot(avg(grad(v_k)), jump(k, n))*avg(r)*dS \
            -avg(nutt)*dot(jump(v_k, n), avg(grad(k)))*avg(r)*dS \
            +dot(jump(v_k), un('+')*k('+') - un('-')*k('-') )*avg(r)*dS  \
            +dot(v_k, un*k)*r*ds \
            -P_*v_k*r*dx() + k*e_/k_ *v_k*r*dx(),           
         
        "e": dot(grad(v_e), nute*grad(e) - u_*e)*r*dx \
            +avg(nute)*(alpha/h('+'))*dot(jump(v_e, n), jump(e, n))*avg(r)*dS 
            -avg(nute)*dot(avg(grad(v_e)), jump(e, n))*avg(r)*dS \
            -avg(nute)*dot(jump(v_e, n), avg(grad(e)))*avg(r)*dS \
            +dot(jump(v_e), un('+')*e('+') - un('-')*e('-') )*avg(r)*dS  \
            +dot(v_e, un*e)*r*ds \
            - (Ce1*P_ - Ce2*e)*e_/k_*v_e*r*dx()
            
         }
else:
    Fke = { "k": nutt*inner(grad(vvk), grad(k))*r*dx() \
             + inner(vvk, dot(grad(k), u_))*r*dx() \
             - Pr*vvk*dx() + k*e_/k_*vvk*r*dx(),
         
        "e": (nu + nut_*(1./sigma_e))*inner(grad(vve), grad(e))*r*dx() \
             + inner(vve, dot(grad(e), u_))*r*dx() \
             - (Ce1*Pr - Ce2*e*r)*e_/k_*vve*dx(),
         
        "ke": nutt*inner(grad(vvk), grad(k))*r*dx() \
             + inner(vvk, dot(grad(k), u_))*r*dx() \
             - Pr*vvk*dx() + e*vvk*r*dx() \
             + (nu + nut_*(1./sigma_e))*inner(grad(vve), grad(e))*r*dx() \
             + inner(vve, dot(grad(e), u_))*r*dx() \
             - (Ce1*Pr - Ce2*e*r)*e_/k_*vve*dx()
         }
        
A = defaultdict(lambda : Matrix())
b = defaultdict(lambda : Vector())

uin0 = interpolate(uin0, Q)
class k_in(Expression):
    def eval(self, values, x):
        intensity = 0.01 if x[1] < R2 else 0.005
        values[0] = 0.5*pow(uin0(x), 2)*intensity

class e_in(Expression):
    def eval(self, values, x):
        kk = 0.5*pow(uin0(x), 2)*0.01
        if x[1] < R1 and x[0] < 10*DOLFIN_EPS_LARGE:
            values[0] = pow(Cmu(0), 0.75)*pow(kk, 1.5) / (0.001)
        elif x[1] < R2 and x[0] < 10*DOLFIN_EPS_LARGE:
            values[0] = pow(Cmu(0), 0.75)*pow(kk, 1.5) / (0.0025)
        else:
            values[0] = e_min

class ke_in(Expression):
    def eval(self, values, x):
        intensity = 0.01 if x[1] < R2 else 0.005
        values[0] = 0.5*pow(uin0(x), 2)*intensity
        if x[1] < R1 and x[0] < 10*DOLFIN_EPS_LARGE:
            values[1] = pow(Cmu(0), 0.75)*pow(values[0], 1.5) / (0.001)
        elif x[1] < R2 and x[0] < 10*DOLFIN_EPS_LARGE:
            values[1] = pow(Cmu(0), 0.75)*pow(values[0], 1.5) / (0.0025)
        else:
            values[1] = e_min

    def value_shape(self):
        return (2,)

if coupled:
    bcs["ke"] = [DirichletBC(TSpace, ke_in(), inlet, "geometric")]
else:
    bcs["k"] = [DirichletBC(TSpace, kin, inlet, "geometric"), 
                DirichletBC(TSpace, k_min, border, "geometric")]
    bcs["e"] = [DirichletBC(TSpace, ein, inlet, "geometric"), 
                DirichletBC(TSpace, e_min, border, "geometric")]
    
bcs["nut"] = []
bcs["T"] = []
bcs["P"] = []

q_ = {"up": up_, "k": k_, "e": e_, "nut": nut_, "P": P_, "T": T_}
x_ = {"up":up_.vector(), "nut": nut_.vector(), "P": P_.vector(), "T": T_.vector()}
if coupled:
    x_.update({"ke": ke_.vector()})
    q_["ke"] = ke_
else:
    x_.update({"k": k_.vector(), "e": e_.vector()})

ke_sol = LUSolver("mumps")
ke_sol.parameters["same_nonzero_pattern"] = True
    
def velocity_iter_Newton(om=-1):
    # Newton iteration for steady flow
    assemble(J, tensor=A["up"])
    for bc in bcs["up"]:
        bc.apply(A["up"])

    om = omega["up"] if om==-1 else om
    func = Fun_cache[VQ]
    func.vector().zero()
    up_sol.solve(A["up"], func.vector(), b["up"])
    up_.vector().axpy(-om, func.vector())
    up_.vector().apply("insert")     
    
    assemble(NS, tensor=b["up"])
    for bc in bcs['up']:
        bc.apply(b["up"], up_.vector())

    return b["up"].norm('l2') 

def velocity_iter_Picard(om=-1):
    # Newton iteration for steady flow
    assemble(lhs(NS_P), tensor=A["up"])
    assemble(rhs(NS_P), tensor=b["up"])
    for bc in bcs['up']:
        bc.apply(A["up"], b["up"])

    om = omega["up"] if om==-1 else om
    func = Fun_cache[VQ]
    func.vector().zero()
    up_sol.solve(A["up"], func.vector(), b["up"])
    func.vector().axpy(-1., up_.vector())
    up_.vector().axpy(om, func.vector())    

    return func.vector().norm("l2") / up_.vector().norm("l2")

count=0
def ke_iter_Picard(comp, om=-1):
    # Newton iterations for steady flow
    global count
    
    assemble(lhs(Fke[comp]), tensor=A[comp])
    assemble(rhs(Fke[comp]), tensor=b[comp])
    for bc in bcs[comp]:
        bc.apply(A[comp], b[comp])

    om = omega[comp] if om==-1 else om
    dk = Fun_cache[TSpace]
    dk.vector().zero()
    ke_sol.solve(A[comp], dk.vector(), b[comp])
    bound(dk.vector(), low_lim=limits[comp][0], upp_lim=limits[comp][1])
    dk.vector().axpy(-1., x_[comp])
    x_[comp].axpy(om, dk.vector())    
    
    if comp == "e":
        count += 1
        if count % 10 == 0:
            plot(q_[comp])
            plot(dk, title="Error in e")
            
    return dk.vector().norm("l2") / x_[comp].norm("l2")

# initialize solution
if coupled:
    x_["ke"][TSpace.sub(0).dofmap().dofs()] = k_min
    x_["ke"][TSpace.sub(1).dofmap().dofs()] = e_min
else:
    x_["k"][:] = k_min
    x_["e"][:] = e_min
#x_["nut"][:] = Cmu(0)*k_min*k_min/e_min
x_["nut"][:] = 0.01
x_["T"][:] = k_min/e_min
x_["P"][:] = 0

# For Newton iterations following is required. For Picard it doesn't hurt
b["up"] = assemble(NS)
for bc in bcs["up"]:
    bc.apply(b["up"], up_.vector())

# Apply boundary conditions to get the true solution on the boundary
for name, val in bcs.iteritems():
    for bc in val:
        bc.apply(x_[name])    
            
def iterate(max_iters=100):
    iter = 0
    error = 1
    while iter < max_iters and error > max_error: 
        error = velocity_iter_Picard()
        #error = velocity_iter_Newton()
        #update(P_, P, bc=bcs["P"], regular=False)
        
        if coupled:
            error_k = ke_iter_Picard("ke")
            error_e = 0
            
        else:
            error_k = ke_iter_Picard("k")        
            
            nut_.update()
            
            error_e = ke_iter_Picard("e")
                
        nut_.update()

        iter += 1
        print "{0:5d} {1:2.5e} {2:2.5e} {3:2.5e} {4:2.5e} ".format(iter, error, error_k, error_e, nut_.vector().norm("l2"))
        error = max(error, max(error_k, error_e))

iterate(max_iters)

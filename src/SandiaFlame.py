__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-08-20"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from numpy import array, linspace, log, minimum, maximum
from os import makedirs, path
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
#set_log_active(False)

params = dict(
    R1 = 3.6e-3,  # Inner radius pilot
    R2 = 9.1e-3,  # Outer radius pilot
    H = 0.15,     # Length of mesh
    L = 0.4,      # Height of mesh
    cl4 = 0.002,  # Mesh density (L, 0)
    cl3 = 0.003,  # Mesh density (L, H)
    cl2 = 0.003,  # Mesh density (0, H)
    cl1 = 0.0001, # Mesh density (0, 0)
    case = "D",
    velocity_pressure_model = "CR", #Velocity-pressure function spaces
    beta = 0.015,  # Numerical stability parameter (artificial viscosity = beta * meshsize)
    nu = Constant(1.58e-5),
    max_error = 1e-7,
    coupled = False,
    max_iters = 200,
    sigma = defaultdict(lambda : 0.00075, {"mf": 0.0001, "var": 0.0001}),  # Parameter used to create smooth inlet profiles. Lower value -> sharper Heaviside = less stable
)

# Any parameter may be overloaded on command line. Read them here
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

params.update(commandline_kwargs)

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

# Underrelaxation factors
omega = defaultdict(lambda : 0.8, {"nut": 0.8})

# k-epsilon model coefficients
model_prm = dict(
    Cmu = Constant(0.09),
    Ce1 = Constant(1.6), # modification from 1.44
    Ce2 = Constant(1.92),
    sigma_e = Constant(1.3),
    sigma_k = Constant(1.0),
    e_d = Constant(0.0))

params.update(model_prm)
vars().update(params)      

# Create Expressions for specifying inlet conditions
inletvalues = {
    "u": 
    {
        "C": {"jet": 29.7, # m/s
              "pilot": 6.8,
              "coflow": 0.9},
        
        "D": {"jet": 49.6,
              "pilot": 11.4,
              "coflow": 0.9}
    }[case]
}

inletvalues.update(
{
    "k": {"jet": 0.5*pow(inletvalues["u"]["jet"], 2)*0.01,
          "pilot": 0.5*pow(inletvalues["u"]["pilot"], 2)*0.01,
          "coflow": 0.5*pow(inletvalues["u"]["coflow"], 2)*0.005}
})
    
inletvalues.update(
{
    "e": {"jet": pow(Cmu(0), 0.75)*pow(inletvalues["k"]["jet"], 1.5) / 0.001,
          "pilot": pow(Cmu(0), 0.75)*pow(inletvalues["k"]["pilot"], 1.5) / 0.0025,
          "coflow": pow(Cmu(0), 0.75)*pow(inletvalues["k"]["coflow"], 1.5) / 0.05},

    "mf":  {"jet": 1, "pilot": 0.25  , "coflow": 0},
    
    "var": {"jet": 1, "pilot": 0.0625, "coflow": 0}
})    
    
k_min = inletvalues["k"]["coflow"]
e_min = inletvalues["e"]["coflow"]

# Define inside functions for boundaries
tol = 10*DOLFIN_EPS_LARGE
def jet(x, on_boundary): 
    return on_boundary and  x[1] < R1 + tol and x[0] < tol
  
def pilot(x, on_boundary): 
    return on_boundary and  x[1] > R1 - tol and x[1] < R2 + tol and x[0] < tol

def coflow(x, on_boundary): 
    return on_boundary and  x[1] > R2 - tol and x[0] < tol
      
def inlet(x, on_boundary): 
    return on_boundary and x[0] < tol

def outlet(x, on_boundary): 
    return on_boundary and x[0] > L - tol or x[1] > H - tol

def centerline(x, on_boundary):
    return on_boundary and x[1] < tol

def border(x, on_boundary):
    return on_boundary and x[1] > H - tol

# Set lower and upper limits for computations
limits = {"k": (k_min/100, 500.),
          "e": (e_min/100, 1e7),
          "nut": (1e-6, 0.01),
          "T": (1e-6, 1e3),
          "ke": (k_min/100, 1e7),
          "mf": (0, 1),
          "var": (0, 1)}

# Define relevant function spaces
Q = FunctionSpace(mesh, "CG", 1)
DG0 = FunctionSpace(mesh, "DG", 0)
DG1 = FunctionSpace(mesh, "DG", 1)
if velocity_pressure_model == "TH":
    V = VectorFunctionSpace(mesh, "CG", 2)
    S = Q
elif velocity_pressure_model == "CR":
    V = VectorFunctionSpace(mesh, "CR", 1)
    S = DG0
    
# Mixed function space for velocity and pressure
VQ = MixedFunctionSpace([V, S])

# Cylindrical coordinate
r = Expression("x[1]", domain=mesh)
r_inv = Expression("x[1] > 1e-8 ? 1.0/x[1] : 0.0", domain=mesh)

n = FacetNormal(mesh)
h = CellSize(mesh)

up = TrialFunction(VQ)
vq = TestFunction(VQ)
v, q = split(vq)
u, p = split(up)

up_ = Function(VQ, name="up")
u_, p_ = split(up_)

# Turbulence
if not coupled:
    # Space for k and epsilon
    TSpace = Q
    k = TrialFunction(TSpace)
    v_k = TestFunction(TSpace)
    e = TrialFunction(TSpace)
    v_e = TestFunction(TSpace)
    k_ = Function(TSpace, name="k")
    e_ = Function(TSpace, name="e")

else:
    # Space for k and epsilon
    TSpace = Q * Q
    k, e = TrialFunctions(TSpace)
    v_k, v_e = TestFunctions(TSpace)
    ke_ = Function(TSpace, name="ke")
    k_, e_ = split(ke_)
    
# Mixture fraction mean and variance
MFSpace = Q
mf = TrialFunction(MFSpace)
v_mf = TestFunction(MFSpace)
mf_ = Function(MFSpace, name="mf")
var_ = Function(MFSpace, name="var")

# This string gives a profile that looks like a combination of smoothed Haviside functions
def smooth(comp):
    ss = "{2}+0.5*(1.0+erf((R1-x[1])/sigma))*({0}-{1}) + 0.5*(1+erf((R2-x[1])/sigma))*({1}-{2})"
    return ss.format(inletvalues[comp]["jet"], inletvalues[comp]["pilot"], inletvalues[comp]["coflow"])

# Create smoothed Expressions for all components on inlet
inlet_Exp = {comp: Expression(smooth(comp), R1=R1, R2=R2, sigma=sigma[comp] ) for comp in ("k", "e", "mf", "var")}
inlet_Exp["ke"] = Expression((smooth("k"), smooth("e")), R1=R1, R2=R2, sigma=sigma["ke"] )
inlet_Exp["u"] = Expression((smooth("u"), "0"), R1=R1, R2=R2, sigma=sigma["u"])
    
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

q_ = {"up": up_, "nut": nut_, "mf": mf_, "var": var_}
x_ = {"up": up_.vector(), "nut": nut_.vector(), "P": P_.vector(), "T": T_.vector(), 
      "mf": mf_.vector(), "var": var_.vector()}
if coupled:
    x_.update({"ke": ke_.vector()})
    q_["ke"] = ke_
else:
    x_.update({"k": k_.vector(), "e": e_.vector()})
    q_.update({"k": k_, "e": e_})

# Create effective viscosity - the sum of viscosity and turbulent viscosity
# A minimum of CellSize is added for stability. This mimicks the effect of upwinding
nutt = nu + nut_ + CellSize(mesh)*Constant(beta)
nute = nu + nut_*(1./sigma_e) + CellSize(mesh)*Constant(beta)

#vv = v + h*dot(grad(v), u_)
vv = v

# Variational form of Navier-Stokes in cylinder coordinates
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

Fke = { 
    "k": nutt*inner(grad(vvk), grad(k))*r*dx() \
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
    
Fmfv = { "mf": nutt*inner(grad(mf), grad(v_mf))*r*dx() \
             + inner(v_mf, dot(grad(mf), u_))*r*dx(),
         
         "var": nutt*inner(grad(mf), grad(v_mf))*r*dx() \
             + inner(v_mf, dot(grad(mf), u_))*r*dx() \
             + 2*e_/k_*v_mf*(mf - mf_*mf_)*r*dx()
}    

A = defaultdict(lambda : Matrix())
b = defaultdict(lambda : Vector())

#Create boundary conditions
bcs = defaultdict(lambda : [], {
    "up":  [DirichletBC(VQ.sub(0), inlet_Exp["u"], inlet),
            DirichletBC(VQ.sub(0).sub(1), 0, centerline),
            DirichletBC(VQ.sub(0).sub(1), 0, border)],
        
    "mf":  [DirichletBC(MFSpace, inlet_Exp["mf"], inlet, "geometric"), 
            DirichletBC(MFSpace, 0, border, "geometric")],
    
    "var": [DirichletBC(MFSpace, inlet_Exp["var"], inlet, "geometric"), 
            DirichletBC(MFSpace, 0, border, "geometric")]
})

if coupled:
    bcs["ke"] =  [DirichletBC(TSpace, inlet_Exp["ke"], inlet, "geometric"),
                  DirichletBC(TSpace, (k_min, e_min), border, "geometric")]
    
else:    
    bcs["k"]  =  [DirichletBC(TSpace, inlet_Exp["k"], inlet, "geometric"), 
                  DirichletBC(TSpace, k_min, border, "geometric")]
    
    bcs["e"]  =  [DirichletBC(TSpace, inlet_Exp["e"], inlet, "geometric"), 
                  DirichletBC(TSpace, e_min, border, "geometric")]
    

parameters["lu_solver"]["same_nonzero_pattern"] = True
solver = defaultdict(lambda : LUSolver("mumps"))
#solver["k"] = KrylovSolver("bicgstab", "ilu")
#solver["k"].parameters["nonzero_initial_guess"] = True
#solver["k"].parameters["monitor_convergence"] = True
#solver["k"].parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
    
def velocity_iter_Newton(om=-1):
    # Newton iteration for steady flow
    assemble(J, tensor=A["up"])
    for bc in bcs["up"]:
        bc.apply(A["up"])

    om = omega["up"] if om==-1 else om
    func = Fun_cache[VQ]
    func.vector().zero()
    solver["up"].solve(A["up"], func.vector(), b["up"])
    up_.vector().axpy(-om, func.vector())
    up_.vector().apply("insert")     
    
    assemble(NS, tnsor=b["up"])
    for bc in bcs["up"]:
        bc.apply(b["up"], up_.vector())

    return b["up"].norm('l2') 

def velocity_iter_Picard(om=-1):
    # Newton iteration for steady flow
    timer = Timer("Velocity/Pressure")
    assemble(lhs(NS_P), tensor=A["up"])
    assemble(rhs(NS_P), tensor=b["up"])
    
    for bc in bcs['up']:
        bc.apply(A["up"], b["up"])

    om = omega["up"] if om==-1 else om
    func = Fun_cache[VQ]
    func.vector().zero()
    solver["up"].solve(A["up"], func.vector(), b["up"])
    func.vector().axpy(-1., up_.vector())
    up_.vector().axpy(om, func.vector())    

    return func.vector().norm("l2") / up_.vector().norm("l2")

count=0
def ke_iter_Picard(comp, om=-1):
    # Newton iterations for steady flow
    global count
    timer = Timer("Turbulence {}".format(comp))
    
    assemble(lhs(Fke[comp]), tensor=A[comp])
    assemble(rhs(Fke[comp]), tensor=b[comp])
    for bc in bcs[comp]:
        bc.apply(A[comp], b[comp])

    om = omega[comp] if om==-1 else om
    dk = Fun_cache[TSpace]
    dk.vector()[:] = x_[comp]
    solver["ke"].solve(A[comp], dk.vector(), b[comp])
    bound(dk.vector(), low_lim=limits[comp][0], upp_lim=limits[comp][1])
    dk.vector().axpy(-1., x_[comp])
    x_[comp].axpy(om, dk.vector())    
    
    if comp == "e":
        count += 1
        if count % 10 == 0:
            plot(q_[comp])
            plot(dk, title="Error in e")
            
    return dk.vector().norm("l2") / x_[comp].norm("l2")

def mfv_solution():
    # Solve mixture fraction equations (linear and thus noniterative)
    assemble(lhs(Fmfv["mf"]), tensor=A["mf"])
    work = Vector(x_["mf"])
    for bc in bcs["mf"]:
        bc.apply(A["mf"], work)

    solver["mf"].solve(A["mf"], x_["mf"], work)
    bound(mf_.vector(), 0, 1)
    
    assemble(lhs(Fmfv["var"]), tensor=A["mf"])
    assemble(rhs(Fmfv["var"]), tensor=b["var"])
    work.zero()
    for bc in bcs["var"]:
        bc.apply(A["mf"], b["var"])
    solver["mf"].solve(A["mf"], x_["var"], b["var"])
    bound(var_.vector(), 0, 1)

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
assemble(NS, tensor=b["up"])
for bc in bcs["up"]:
    bc.apply(b["up"], up_.vector())

# Apply boundary conditions to get the true solution on the boundary
for name, val in bcs.iteritems():
    for bc in val:
        bc.apply(x_[name])    

def store_solution(for_plotting=False):
    if for_plotting == False:
        foldername = "../results/{}".format(V.dim())
        try:
            makedirs(foldername)
        except OSError:
            pass
        
        for key, val in q_.iteritems():
            h5file = path.join(foldername, key+".h5")
            newfile = HDF5File(mpi_comm_world(), h5file, "w")
            newfile.write(val, val.name())    
    else:
        foldername = "../VTK/{}".format(V.dim())
        try:
            makedirs(foldername)
        except OSError:
            pass
        
        for key, val in q_.iteritems():
            if not key == "up":
                h5file = path.join(foldername, key+".xdmf")
                newfile = XDMFFile(mpi_comm_world(), h5file)
                newfile << (val, 0.0)
            else:
                h5file = path.join(foldername, "u.xdmf")
                newfile = XDMFFile(mpi_comm_world(), h5file)
                newfile << (Function(project(val.sub(0), V), name="u"), 0.0)
                h5file = path.join(foldername, "p.xdmf")
                newfile = XDMFFile(mpi_comm_world(), h5file)
                newfile << (Function(project(val.sub(1), S), name="p"), 0.0)

def read_solution():
    foldername = "../results/{}".format(V.dim())
    if not path.exists(foldername):        
        raise IOError(foldername+" does not exist")
        return
        
    for h5file in os.listdir(foldername):
        newfile = HDF5File(mpi_comm_world(), path.join(foldername, h5file), "r")
        try:
            newfile.read(q_[h5file[:-3]], h5file[:-3])
        except:
            pass
            
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

def iterate_mfv(max_iters=100):
    iter = 0
    error = 1
    while iter < max_iters and error > max_error: 
        error = mfv_iter_Newton()
        iter += 1
        print "{0:5d} {1:2.5e}".format(iter, error)    
    
iterate(max_iters)
#read_solution()
mfv_solution()
store_solution(True)


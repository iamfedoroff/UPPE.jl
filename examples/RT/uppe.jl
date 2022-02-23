using UPPE
using UnPack

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val

cd(@__DIR__)

include("air.jl")
UPPE.permittivity(w) = permittivity(w)
UPPE.permeability(w) = permeability(w)


# ******************************************************************************
grid = GridRT(
    GPU();
    ru = 1e-3,   # [m] unit of space in r direction
    rmax = 10,   # [ru] area in spatial domain
    Nr = 500,   # number of points in spatial domain
    rguard = 1,   # [ru] the width of the lossy slab at the end of r grid
    kguard = 45,   # [degrees] the cut-off angle for wave vectors
    tu = 1e-15,   # [s] unit of time
    tmin = -200,   # [tu] left boundary of t grid
    tmax = 200,   # [tu] right boundary of t grid
    Nt = 2^11,   # number of points on t grid
    tguard = 20,   # [tu] the width of the lossy slab at the ends of t grid
    wguard = 1e16,   # [1/s] the cut-off angular frequency
)


# ******************************************************************************
lam0 = 0.8e-6   # [m] central wavelength
a0 = 2e-3 / (2 * sqrt(log(2)))  # [m] initial beam radius
tau0 = 35e-15 / (2 * sqrt(log(2)))  # [s] initial pulse duration
W = 2e-3   # [J] initial pulse energy

w0 = 2 * pi * C0 / lam0   # central frequency
I0 = W / (pi^1.5 * tau0 * a0^2)   # initial pulse intensity
Iu = I0   # [W/m**2] unit of intensity
n0 = refractive_index(w0)
Eu = intensity2field(Iu, n0)

E = zeros((grid.Nr, grid.Nt))
for it=1:grid.Nt
for ir=1:grid.Nr
    E[ir,it] = sqrt(I0 / Iu) *
               exp(-0.5 * (grid.r[ir] * grid.ru)^2 / a0^2) *
               exp(-0.5 * (grid.t[it] * grid.tu)^2 / tau0^2) *
               cos(w0 * grid.t[it] * grid.tu)
end
end
field = Field(grid, w0, E; Eu, Iu)


# ******************************************************************************
function kerr_response_func!(ne, Pnl, J, E, p, z)
    Rk, = p
    @. Pnl = Rk * real(E)^3
    return nothing
end


function kerr_response(grid::Grid{T}, field, n2) where T
    @unpack arch = grid
    @unpack Eu, w0, E = field

    chi3 = chi3_func(w0, n2)
    Rk = EPS0 * chi3 * Eu^3
    p_resp = (Rk, )

    # response arrays:
    ne = adapt_array(arch, zeros(size(E)))
    Pnl = zero(E)
    J = zero(E)

    TF = typeof(kerr_response_func!)
    TP = typeof(p_resp)
    return Response{T, TF, TP}(ne, Pnl, J, kerr_response_func!, p_resp)
end


response =  kerr_response(grid, field, n2)

model = Model(
    grid, field, response;
    zu = 1,   # [m] unit of space in z direction
    z = 0,   # [m] initial propagation distance
    zmax = 5,   # [zu] propagation distance
    kparaxial = true,   # paraxial approximation for the linear term
    qparaxial = true,   # paraxial approximation for the nonlinear term
    nonlinearity = true,   # switch for the nonlinear term
)

simulation = Simulation(
    model, n2, N0;
    prefix = "results/",
    dz0 = 0.01,   # [zu] initial z step
    dzout = 0.5,   # [zu] z step for writing field into output file
    phimax = pi/200,  # maximum nonlinear phase increment for adaptive z step
    Icrit = 10,   # [Iu] maixmum allowed intensity (stop if exceeded)
    alg = RK4(),   # Algorithm of ODE solver for nonlinear part
)

run!(simulation)

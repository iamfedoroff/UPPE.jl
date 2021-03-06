module UPPE

export CPU, GPU, adapt_array,
       Grid, GridT, GridRT,
       Field, field2intensity, intensity2field,
       refractive_index, k_func, chi3_func, group_velocity, diffraction_length,
       dispersion_length,
       Response, Model,
       Simulation, run!,
       RK2, RK3, RK4, Tsit5, ATsit5   # ODEIntegrators algorithms

using AnalyticSignals: rsig2asig!, rsig2aspec!
using CUDA: CuVector, CuMatrix, CuArray, @cuda, launch_configuration,
            threadIdx, blockIdx, blockDim, gridDim
using FFTW: fftfreq, ifftshift, plan_fft!
using HankelTransforms: dhtcoord, dhtfreq, plan_dht
using HDF5
using ODEIntegrators
using Printf: @printf, @sprintf
using TimerOutputs: @timeit, reset_timer!, get_defaulttimer

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val
const EPS0 = VacuumElectricPermittivity.val
const MU0 = VacuumMagneticPermeability.val
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val

abstract type ARCH{T} end
struct CPU{T} <: ARCH{T} end
struct GPU{T} <: ARCH{T} end
CPU() = CPU{Float64}()
GPU() = GPU{Float32}()

float_type(::ARCH{T}) where T = T

adapt_array(::CPU, a::AbstractArray) = a
adapt_array(::GPU, a::AbstractArray) = CuArray(a)

include("grids.jl")
include("fields.jl")
include("medium.jl")
include("guards.jl")
include("models.jl")
include("field_analyzers.jl")
include("outputs.jl")
include("simulations.jl")

end

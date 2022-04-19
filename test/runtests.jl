using UPPE
using Test

using PhysicalConstants.CODATA2018
const C0 = SpeedOfLightInVacuum.val

cd(@__DIR__)

@testset "UPPE.jl" begin
    include("test_t.jl")
    include("test_rt.jl")
end

rm("results", recursive=true)

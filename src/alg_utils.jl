############################################################################################
# For the equivalent parts in OrdinaryDiffEq.jl, see:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/src/alg_utils.jl
############################################################################################

OrdinaryDiffEq.alg_autodiff(alg::AbstractEK) = true
OrdinaryDiffEq.get_chunksize(alg::AbstractEK) = Val(0)
OrdinaryDiffEq.isfsal(alg::AbstractEK) = false


############################################
# Step size control
OrdinaryDiffEq.isadaptive(alg::AbstractEK) = true
OrdinaryDiffEq.alg_order(alg::AbstractEK) =
    alg.errest == :embedded ? alg.emb_order+1 : alg.order+1
# OrdinaryDiffEq.alg_adaptive_order(alg::AbstractEK) =

# PI control is the default!
OrdinaryDiffEq.isstandard(alg::AbstractEK) = false # proportional
OrdinaryDiffEq.ispredictive(alg::AbstractEK) = false # not sure, maybe Gustafsson acceleration?

# OrdinaryDiffEq.qmin_default(alg::AbstractEK) =
# OrdinaryDiffEq.qmax_default(alg::AbstractEK) =
OrdinaryDiffEq.beta2_default(alg::AbstractEK) = 2//(5(alg.order+1))
OrdinaryDiffEq.beta1_default(alg::AbstractEK, beta2) = 7//(10(alg.order+1))
# OrdinaryDiffEq.gamma_default(alg::AbstractEK) =

# OrdinaryDiffEq.uses_uprev(alg::, adaptive::Bool) = adaptive
OrdinaryDiffEq.is_mass_matrix_alg(alg::AbstractEK) = true




OrdinaryDiffEq.isadaptive(alg::DAE_EK1) = true
OrdinaryDiffEq.alg_order(alg::DAE_EK1) = alg.order+1
OrdinaryDiffEq.alg_autodiff(alg::DAE_EK1) = true
OrdinaryDiffEq.get_chunksize(alg::DAE_EK1) = Val(0)
OrdinaryDiffEq.isfsal(alg::DAE_EK1) = false


OrdinaryDiffEq.gamma_default(alg::DAE_EK1) = 8//10

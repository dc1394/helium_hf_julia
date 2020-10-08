include("helium_hf_julia.jl")
using Printf
using .Helium_HF

function main()
    res = Helium_HF.do_scfloop()
    if (res != nothing) 
        @printf "SCF計算が収束しました: energy = %.14f (Hartree)\n" res
    else
        @printf "SCF計算が収束しませんでした\n"
    end
end

main()
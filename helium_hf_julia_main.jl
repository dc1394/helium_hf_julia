include("helium_hf_julia.jl")
using Printf
using .Helium_HF

const RESULT_FILENAME = "result.csv"

function main()
    res = Helium_HF.do_scfloop()
    if (res != nothing)
        alpha, c, energy = res
        Helium_HF.save_result(alpha, c, RESULT_FILENAME)
        @printf "SCF計算が収束しました: energy = %.14f (Hartree)、計算結果を%sに書き込みました\n" energy RESULT_FILENAME
    else
        @printf "SCF計算が収束しませんでした\n"
    end
end

main()
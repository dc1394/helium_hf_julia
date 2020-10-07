module Helium_HF
    using LinearAlgebra
    using Match
    using Printf

    const MAXBUFSIZE = 32

    const MAXITER = 1000

    const SCFTHRESHOLD = 1.0E-15

    function do_scfloop()
        # 使用するGTOの数を入力
        nalpha = 6 #input_nalpha()

        # GTOの肩の係数が格納された配列を生成
        alpha = make_alpha(nalpha)

        # 1電子積分が格納された2次元配列を生成
        h = make_oneelectroninteg(alpha)

        # 2電子積分が格納された4次元配列を生成
        q = make_twoelectroninteg(alpha)

        # 重なり行列を生成
        s = make_overlapmatrix(alpha)

        # 全て0.0で初期化された固有ベクトルを生成
        c = make_c(nalpha, 0.0)

        # 新しく計算されたエネルギー
        enew = 0.0;

        # SCFループ
        for iter = 1:MAXITER
            # Fock行列を生成
            f = make_fockmatrix(c, h, q)

            # 一般化固有値問題を解く
            eigenval, eigenvec = eigen(f, s)
            
            # E'を取得
            ep = eigenval[1]
            
            # 固有ベクトルを取得
            c = vec(eigenvec)[1:nalpha]

            # 固有ベクトルを正規化
            c = normalize(alpha, c)

            # 前回のSCF計算のエネルギーを保管
            eold = enew;

            # 今回のSCF計算のエネルギーを計算する
            enew = getenergy(c, ep, h);

            @printf "Iteration # %2d: HF eigenvalue = %.14f, energy = %.14f\n" iter ep enew

            # SCF計算が収束したかどうか
            if abs(enew - eold) < SCFTHRESHOLD 
                # 収束したのでそのエネルギーを返す
                return enew
            end
        end     
       
        # SCF計算が収束しなかった
        return nothing
    end

    function getenergy(c, ep, h)
        nalpha = length(c)
        e = ep
        
        for p = 1:nalpha
            for q = 1:nalpha
                # E = E' + Cp * Cq * hpq
                e += c[p] * c[q] * h[p, q]
            end
        end

        return e
    end

    function input_nalpha()
        nalpha = 0

        while true
            @printf "使用するGTOの個数を入力してください (3, 4 or 6): "
            s = readline()

            nalpha = tryparse(Int64, s)
            if nalpha != nothing
                @match nalpha begin
                    3 => break
                    4 => break
                    6 => break
                    _ => continue
                end
            end
        end

        return nalpha
    end

    function make_alpha(nalpha)
        alpha = @match nalpha begin
                    3 => [0.31364978999999998, 1.1589229999999999, 6.3624213899999997]
                    4 => [0.297104, 1.236745, 5.749982, 38.2166777]
                    6 => [0.18595935599999999, 0.45151632200000003, 1.1627151630000001, 3.384639924, 12.09819836, 65.984568240000002]
                    _ => []
                end
        return alpha
    end

    function make_c(nalpha, val)
        return [val for i = 1:nalpha]
    end

    function make_fockmatrix(c, h, q)
        nalpha = length(c)
        f = zeros(nalpha, nalpha)

        for p = 1:nalpha
            for qi = 1:nalpha
                # Fpq = hpq + ΣCr * Cs * Qprqs
                f[p, qi] = h[p, qi];

                for r = 1:nalpha
                    for s = 1:nalpha
                        f[p, qi] += c[r] * c[s] * q[p, r, qi, s]
                    end
                end
            end
        end

        return f
    end

    function make_oneelectroninteg(alpha)
        nalpha = length(alpha);
        h =  zeros(nalpha, nalpha)

        for p = 1:nalpha
            for q = 1:nalpha
                # αp + αq
                appaq = alpha[p] + alpha[q];

                # hpq = 3αpαqπ^1.5 / (αp + αq)^2.5 - 4π / (αp + αq)
                h[p, q] = 3.0 * alpha[p] * alpha[q] * ((pi / appaq) ^ 1.5) / appaq -
                          4.0 * pi / appaq
            end
        end

        return h
    end
    
    function make_overlapmatrix(alpha)
        nalpha = length(alpha)
        s = zeros(nalpha, nalpha)

        for p = 1:nalpha
            for q = 1:nalpha
                # Spq = (π / (αp + αq))^1.5
                s[p, q] = (pi / (alpha[p] + alpha[q])) ^ 1.5
            end
        end

        return s
    end
    
    function make_twoelectroninteg(alpha)
        nalpha = length(alpha)
        q = zeros(nalpha, nalpha, nalpha, nalpha)

        for p = 1:nalpha
            for qi = 1:nalpha
                for r = 1:nalpha
                    for s = 1:nalpha
                        # Qprqs = 2π^2.5 / [(αp + αq)(αr + αs)√(αp + αq + αr + αs)]
                        q[p, r, qi, s] = 2.0 * (pi ^ 2.5) /
                            ((alpha[p] + alpha[qi]) * (alpha[r] + alpha[s]) *
                            sqrt(alpha[p] + alpha[qi] + alpha[r] + alpha[s]))
                    end
                end
            end
        end

        return q
    end

    function normalize(alpha, c)
        nalpha = length(c)

        sum = 0.0
        for i = 1:nalpha
            for j = 1:nalpha
                sum += c[i] * c[j] / (4.0 * (alpha[i] + alpha[j])) * (pi / (alpha[i] + alpha[j])) ^ 0.5
            end
        end 

        return map(item -> item / sqrt(4.0 * pi * sum), c)
    end
end
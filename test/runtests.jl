using Test

# Assume que os testes são executados a partir do diretório raiz do projeto.
# Adiciona o diretório `genetic` ao caminho para que o `include` funcione.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "genetic"))

include("Utils.jl")
using .UtilsModule

@testset "Validação de Soluções TSP - Utils.jl" begin
    @testset "Casos Válidos" begin
        @test is_valid_solution([1, 2, 3, 4], 4)
        @test is_valid_solution([4, 1, 3, 2], 4)
        @test is_valid_solution(collect(1:100), 100)
        @test is_valid_solution([1], 1)
    end

    @testset "Casos Inválidos" begin
        # Comprimento incorreto
        @test !is_valid_solution([1, 2, 3], 4)
        @test !is_valid_solution([1, 2, 3, 4, 5], 4)

        # Elementos duplicados
        @test !is_valid_solution([1, 2, 2, 4], 4)

        # Números fora do intervalo
        @test !is_valid_solution([1, 2, 3, 5], 4)
        @test !is_valid_solution([0, 1, 2, 3], 4)

        # Rota vazia
        @test !is_valid_solution(Int[], 4)
        @test !is_valid_solution(Int[], 0) # Válido apenas se n=0
    end
end

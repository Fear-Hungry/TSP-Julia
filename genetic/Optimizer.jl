module OptimizerModule

export GeneticOptimizer, evolve!, initialize_population!

using Random
using Printf
using StatsBase
using Logging
using Distributed
using SparseArrays

# Importa o módulo Individual
include("Individual.jl")
using .IndividualModule

# Importa o módulo AOS
include("AOS.jl")
using .AOSModule

# --- Configuração do Logger para este módulo ---
const logger = SimpleLogger(stdout, Logging.Info)
macro log(level, msg)
    quote
        with_logger(logger) do
            @logmsg($level, $msg)
        end
    end |> esc
end
# ---------------------------------------------

"""
    GeneticOptimizer

Estrutura que encapsula os parâmetros e o estado do algoritmo genético.
"""
mutable struct GeneticOptimizer
    population::Vector{Individual}
    population_size::Int
    elite_size_ratio::Float64
    generations::Int
    n_cities::Int

    # Dados do problema, compartilhados por todos
    coords::Tuple{Vector{Float32}, Vector{Float32}}
    dist_matrix::AbstractMatrix{Float32}

    # Gerenciador de Seleção de Operadores
    aos_manager::Union{UCB1Manager, Nothing}

    # Mapeamento de operadores para suas funções
    operator_functions::Dict{Symbol, Function}

    # Parâmetros para o modo padrão (não-AOS)
    mutation_rate::Float64
    two_opt_rate::Float64
    crossover_rate::Float64
end

"""
    GeneticOptimizer(...)

Construtor para o otimizador genético.
"""
function GeneticOptimizer(;population_size::Int=100, elite_size_ratio::Float64=0.1,
                          generations::Int=100, n_cities::Int,
                          coords::Tuple, dist_matrix::AbstractMatrix,
                          aos_mode::Union{String, Nothing}=nothing, ucb_c::Float64=2.0,
                          mutation_rate::Float64=0.01, two_opt_rate::Float64=0.1, crossover_rate::Float64=0.7)

    operator_functions = create_operator_function_mapping(dist_matrix, coords)
    aos_manager = create_aos_manager_if_needed(aos_mode, operator_functions, ucb_c)

    return GeneticOptimizer(
        Vector{Individual}(), population_size, elite_size_ratio, generations, n_cities,
        coords, dist_matrix, aos_manager, operator_functions,
        mutation_rate, two_opt_rate, crossover_rate
    )
end

function create_operator_function_mapping(dist_matrix, coords)
    return Dict{Symbol, Function}(
        :crossover => (p1, p2) -> perform_crossover(p1, p2, dist_matrix, coords),
        :swap_mutation => (ind) -> apply_swap_mutation!(ind, dist_matrix, coords),
        :inversion_mutation => (ind) -> apply_inversion_mutation!(ind, dist_matrix, coords, rate=1.0),
        :two_opt => (ind) -> apply_two_opt_mutation!(ind, dist_matrix, coords)
    )
end

function create_aos_manager_if_needed(aos_mode, operator_functions, ucb_c)
    if aos_mode == "ucb1"
        operation_names = collect(keys(operator_functions))
        @log Logging.Info "Otimizador criado com UCB1 (c=$(ucb_c))"
        return UCB1Manager(operation_names, exploration_constant=ucb_c)
    end
    return nothing
end

"""
    initialize_population!(optimizer)

Inicializa a população do otimizador com indivíduos aleatórios.
"""
function initialize_population!(optimizer::GeneticOptimizer)
    optimizer.population = Vector{Individual}(undef, optimizer.population_size)

    for i in 1:optimizer.population_size
        route = create_random_tsp_route(optimizer.n_cities)
        fitness = calculate_route_fitness(route, optimizer.dist_matrix, optimizer.coords)
        optimizer.population[i] = Individual(route, fitness)
    end
end

"""
    select_elite(population, elite_ratio) -> Vector{Individual}

Seleciona os melhores indivíduos (elite) de uma população.
"""
function select_elite_individuals(population::Vector{Individual}, elite_ratio::Float64)
    sort!(population, by = individual -> individual.fitness)
    elite_count = round(Int, length(population) * elite_ratio)
    return population[1:elite_count]
end

"""
    _evolve_ucb!(optimizer)

Evolui a população usando a seleção de operadores UCB1.
"""
function evolve_with_ucb1!(optimizer::GeneticOptimizer)
    convergence_history = create_convergence_tracker()

    for generation in 1:optimizer.generations
        process_generation_with_ucb1!(optimizer, generation, convergence_history)
    end

    return extract_best_solution(optimizer.population, convergence_history)
end

function create_convergence_tracker()
    return Dict("generations" => Int[], "distances" => Float32[])
end

function process_generation_with_ucb1!(optimizer::GeneticOptimizer, generation::Int, convergence_history)
    sort!(optimizer.population, by = individual -> individual.fitness)
    record_generation_progress(generation, optimizer, convergence_history)

    elite_individuals = select_elite_individuals(optimizer.population, optimizer.elite_size_ratio)
    new_population = deepcopy(elite_individuals)

    fill_population_with_ucb1!(optimizer, new_population, elite_individuals)
    optimizer.population = new_population
end

function record_generation_progress(generation::Int, optimizer::GeneticOptimizer, convergence_history)
    best_fitness = optimizer.population[1].fitness
    push!(convergence_history["generations"], generation)
    push!(convergence_history["distances"], best_fitness)

    if generation % 100 == 0
        @log Logging.Info @sprintf("Geração: %d/%d | Melhor Distância: %.2f",
                                  generation, optimizer.generations, best_fitness)
    end
end

function fill_population_with_ucb1!(optimizer::GeneticOptimizer, new_population::Vector{Individual}, elite::Vector{Individual})
    while length(new_population) < optimizer.population_size
        child = create_child_with_ucb1(optimizer, elite)
        push!(new_population, child)
    end
end

function create_child_with_ucb1(optimizer::GeneticOptimizer, elite::Vector{Individual})
    selected_operator = select_operator!(optimizer.aos_manager)
    operator_function = optimizer.operator_functions[selected_operator]

    parent1, parent2 = sample(elite, 2, replace=false)

    if selected_operator == :crossover
        child = operator_function(parent1, parent2)
        reward = calculate_crossover_reward(parent1, parent2, child)
    else
        child = apply_mutation_operator(operator_function, parent1)
        reward = calculate_mutation_reward(parent1, child)
    end

    update_stats!(optimizer.aos_manager, selected_operator, reward)
    return child
end

function calculate_crossover_reward(parent1::Individual, parent2::Individual, child::Individual)
    parent_average_fitness = (parent1.fitness + parent2.fitness) / 2
    return parent_average_fitness - child.fitness
end

function apply_mutation_operator(operator_function::Function, parent::Individual)
    mutated_individual = deepcopy(parent)
    original_fitness = mutated_individual.fitness
    operator_function(mutated_individual)
    return mutated_individual
end

function calculate_mutation_reward(original::Individual, mutated::Individual)
    return original.fitness - mutated.fitness
end

"""
    _evolve_sequential!(optimizer)

Evolui a população de forma sequencial com taxas fixas.
"""
function evolve_with_fixed_rates!(optimizer::GeneticOptimizer)
    convergence_history = create_convergence_tracker()

    for generation in 1:optimizer.generations
        process_generation_with_fixed_rates!(optimizer, generation, convergence_history)
    end

    return extract_best_solution(optimizer.population, convergence_history)
end

function process_generation_with_fixed_rates!(optimizer::GeneticOptimizer, generation::Int, convergence_history)
    sort!(optimizer.population, by = individual -> individual.fitness)
    push!(convergence_history["generations"], generation)
    push!(convergence_history["distances"], optimizer.population[1].fitness)

    elite_individuals = select_elite_individuals(optimizer.population, optimizer.elite_size_ratio)
    new_population = deepcopy(elite_individuals)

    fill_population_with_fixed_rates!(optimizer, new_population, elite_individuals)
    optimizer.population = new_population
end

function fill_population_with_fixed_rates!(optimizer::GeneticOptimizer, new_population::Vector{Individual}, elite::Vector{Individual})
    while length(new_population) < optimizer.population_size
        parent1, parent2 = sample(elite, 2, replace=false)

        if should_apply_crossover(optimizer.crossover_rate)
            child = create_child_with_crossover_and_mutation(optimizer, parent1, parent2)
        else
            child = select_random_parent(parent1, parent2)
        end

        push!(new_population, child)
    end
end

function should_apply_crossover(crossover_rate::Float64)
    return rand() < crossover_rate
end

function create_child_with_crossover_and_mutation(optimizer::GeneticOptimizer, parent1::Individual, parent2::Individual)
    child = perform_crossover(parent1, parent2, optimizer.dist_matrix, optimizer.coords)
    apply_mutations!(child, optimizer.dist_matrix, optimizer.coords,
                    swap_rate=optimizer.mutation_rate,
                    two_opt_rate=optimizer.two_opt_rate)
    return child
end

function select_random_parent(parent1::Individual, parent2::Individual)
    return deepcopy(rand() < 0.5 ? parent1 : parent2)
end

function extract_best_solution(population::Vector{Individual}, convergence_history)
    sort!(population, by = individual -> individual.fitness)
    best_individual = population[1]
    return best_individual.route, best_individual.fitness, convergence_history
end

"""
    evolve!(optimizer) -> Tuple

Executa o ciclo de evolução, despachando para a versão correta (AOS ou Padrão).
"""
function evolve!(optimizer::GeneticOptimizer)
    if optimizer.aos_manager !== nothing
        @log Logging.Info "Iniciando evolução com UCB1"
        return evolve_with_ucb1!(optimizer)
    else
        @log Logging.Info "Iniciando evolução com taxas fixas"
        return evolve_with_fixed_rates!(optimizer)
    end
end

end # fim do módulo

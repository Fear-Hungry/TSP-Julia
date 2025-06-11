module IndividualModule

export Individual, create_random_tsp_route, calculate_route_fitness,
       perform_crossover, apply_mutations!, apply_two_opt_improvement!,
       apply_swap_mutation!, apply_inversion_mutation!, apply_two_opt_mutation!

using Random
using StatsBase
using SparseArrays
using Distances

"""
    Individual

Estrutura que representa um indivíduo (uma rota candidata) no algoritmo genético.
Não armazena mais a matriz de distância para economizar memória.
"""
mutable struct Individual
    route::Vector{Int}
    fitness::Float32
    tabu_moves::Vector{Tuple{Int, Int}}
end

"""
    Individual(route, fitness, tabu_moves)

Construtor direto para um `Individual`.
"""
function Individual(route::Vector{Int}, fitness::Float32)
    return Individual(route, fitness, Vector{Tuple{Int, Int}}())
end

"""
    create_random_tsp_route(num_cities::Int) -> Vector{Int}

Cria um caminho (permutação) aleatório de `num_cities` cidades.
"""
function create_random_tsp_route(num_cities::Int)
    route = collect(1:num_cities)
    shuffle!(route)
    return route
end

"""
    calculate_route_fitness(route, distance_matrix, coordinates) -> Float32

Calcula o fitness (distância total) de um gene.
"""
function calculate_route_fitness(route::Vector{Int}, distance_matrix::AbstractMatrix, coordinates)
    total_distance = 0.0f0
    route_length = length(route)

    for i in 1:(route_length - 1)
        total_distance += get_city_distance(distance_matrix, coordinates, route[i], route[i+1])
    end

    total_distance += get_city_distance(distance_matrix, coordinates, route[route_length], route[1])
    return total_distance
end

"""
    get_city_distance(distance_matrix, coords, i, j) -> Float32

Obtém a distância entre as cidades `i` e `j`. Tenta usar a matriz (esparsa) como cache.
Se a distância não estiver na matriz, calcula on-the-fly.
"""
@inline function get_city_distance(distance_matrix::AbstractMatrix, coords::Tuple{Vector{Float32}, Vector{Float32}}, city1::Int, city2::Int)
    cached_distance = distance_matrix[city1, city2]

    return cached_distance > 0 ? cached_distance : calculate_euclidean_distance(coords, city1, city2)
end

function calculate_euclidean_distance(coords::Tuple{Vector{Float32}, Vector{Float32}}, city1::Int, city2::Int)
    x_coords, y_coords = coords
    return euclidean((x_coords[city1], y_coords[city1]), (x_coords[city2], y_coords[city2]))
end

"""
    perform_crossover(parent1, parent2, dist_matrix, coords) -> Individual

Realiza o Ordered Crossover (OX1) entre dois pais para gerar um filho.
"""
function perform_crossover(parent1::Individual, parent2::Individual, distance_matrix::AbstractMatrix, coordinates)
    child_route = create_ordered_crossover(parent1.route, parent2.route)
    child_fitness = calculate_route_fitness(child_route, distance_matrix, coordinates)

    return Individual(child_route, child_fitness, copy(parent1.tabu_moves))
end

function create_ordered_crossover(parent1_route::Vector{Int}, parent2_route::Vector{Int})
    route_length = length(parent1_route)
    child_route = zeros(Int, route_length)

    cut_point1, cut_point2 = select_crossover_points(route_length)
    copy_parent_segment!(child_route, parent1_route, cut_point1, cut_point2)
    fill_remaining_cities!(child_route, parent2_route, cut_point1, cut_point2)

    return child_route
end

function select_crossover_points(route_length::Int)
    return sort(sample(1:route_length, 2, replace=false))
end

function copy_parent_segment!(child_route::Vector{Int}, parent_route::Vector{Int}, start_pos::Int, end_pos::Int)
    parent_segment = parent_route[start_pos:end_pos]
    child_route[start_pos:end_pos] = parent_segment
    return Set(parent_segment)
end

function fill_remaining_cities!(child_route::Vector{Int}, parent2_route::Vector{Int}, cut_point1::Int, cut_point2::Int)
    used_cities = Set(child_route[cut_point1:cut_point2])
    child_position = 1
    parent2_position = 1

    while child_position <= length(child_route)
        if child_position >= cut_point1 && child_position <= cut_point2
            child_position += 1
            continue
        end

        city = find_next_unused_city(parent2_route, parent2_position, used_cities)
        child_route[child_position] = city
        parent2_position = findfirst(==(city), parent2_route) + 1
        child_position += 1
    end
end

function find_next_unused_city(parent_route::Vector{Int}, start_position::Int, used_cities::Set{Int})
    for i in start_position:length(parent_route)
        if parent_route[i] ∉ used_cities
            return parent_route[i]
        end
    end
end

"""
    apply_two_opt_improvement!(individual, dist_matrix, coords) -> Bool

Aplica a heurística 2-opt para melhorar um indivíduo, modificando-o no local.
"""
function apply_two_opt_improvement!(individual::Individual, distance_matrix::AbstractMatrix, coordinates)
    route_length = length(individual.route)
    improvement_found = false

    for i in 1:(route_length-2)
        for j in (i+2):route_length
            if can_improve_with_two_opt(individual, distance_matrix, coordinates, i, j)
                apply_two_opt_swap!(individual, i, j, distance_matrix, coordinates)
                improvement_found = true
            end
        end
    end

    return improvement_found
end

function can_improve_with_two_opt(individual::Individual, distance_matrix::AbstractMatrix, coordinates, i::Int, j::Int)
    i_next = i + 1
    j_next = (j % length(individual.route)) + 1

    current_distance = calculate_edge_pair_distance(individual.route, distance_matrix, coordinates, i, i_next, j, j_next)
    new_distance = calculate_edge_pair_distance(individual.route, distance_matrix, coordinates, i, j, i_next, j_next)

    return new_distance < current_distance - 1e-9
end

function calculate_edge_pair_distance(route::Vector{Int}, distance_matrix::AbstractMatrix, coordinates, city1_idx::Int, city2_idx::Int, city3_idx::Int, city4_idx::Int)
    city1, city2 = route[city1_idx], route[city2_idx]
    city3, city4 = route[city3_idx], route[city4_idx]

    return get_city_distance(distance_matrix, coordinates, city1, city2) +
           get_city_distance(distance_matrix, coordinates, city3, city4)
end

function apply_two_opt_swap!(individual::Individual, i::Int, j::Int, distance_matrix::AbstractMatrix, coordinates)
    current_distance = calculate_current_swap_distance(individual, i, j, distance_matrix, coordinates)
    new_distance = calculate_new_swap_distance(individual, i, j, distance_matrix, coordinates)

    reverse!(@view(individual.route[(i+1):j]))
    individual.fitness += (new_distance - current_distance)
end

function calculate_current_swap_distance(individual::Individual, i::Int, j::Int, distance_matrix::AbstractMatrix, coordinates)
    i_next = i + 1
    j_next = (j % length(individual.route)) + 1

    city_i, city_i_next = individual.route[i], individual.route[i_next]
    city_j, city_j_next = individual.route[j], individual.route[j_next]

    return get_city_distance(distance_matrix, coordinates, city_i, city_i_next) +
           get_city_distance(distance_matrix, coordinates, city_j, city_j_next)
end

function calculate_new_swap_distance(individual::Individual, i::Int, j::Int, distance_matrix::AbstractMatrix, coordinates)
    i_next = i + 1
    j_next = (j % length(individual.route)) + 1

    city_i, city_j = individual.route[i], individual.route[j]
    city_i_next, city_j_next = individual.route[i_next], individual.route[j_next]

    return get_city_distance(distance_matrix, coordinates, city_i, city_j) +
           get_city_distance(distance_matrix, coordinates, city_i_next, city_j_next)
end

"""
    apply_mutations!(individual, dist_matrix, coords; swap_rate, two_opt_rate)

Aplica mutações (swap e/ou 2-opt) a um indivíduo no local.
"""
function apply_mutations!(individual::Individual, distance_matrix::AbstractMatrix, coordinates; swap_rate::Float64 = 0.01, two_opt_rate::Float64 = 0.1)
    mutations_applied = false

    if should_apply_mutation(swap_rate)
        mutations_applied |= apply_swap_mutation!(individual, distance_matrix, coordinates)
    end

    if should_apply_mutation(two_opt_rate)
        mutations_applied |= apply_two_opt_mutation!(individual, distance_matrix, coordinates)
    end

    return mutations_applied
end

function should_apply_mutation(mutation_rate::Float64)
    return rand() < mutation_rate
end

"""
    apply_swap_mutation!(individual, dist_matrix, coords)

Aplica mutação por troca (swap).
"""
function apply_swap_mutation!(individual::Individual, distance_matrix::AbstractMatrix, coordinates)
    route_length = length(individual.route)
    num_swaps = calculate_num_swaps(route_length)
    swap_positions = select_random_positions(route_length, num_swaps * 2)

    swaps_performed = perform_valid_swaps!(individual, swap_positions)

    if swaps_performed
        individual.fitness = calculate_route_fitness(individual.route, distance_matrix, coordinates)
    end

    return swaps_performed
end

function calculate_num_swaps(route_length::Int)
    return max(1, round(Int, route_length * 0.005))
end

function select_random_positions(route_length::Int, num_positions::Int)
    return sample(1:route_length, num_positions, replace=false)
end

function perform_valid_swaps!(individual::Individual, swap_positions::Vector{Int})
    swaps_performed = false

    for i in 1:2:(length(swap_positions))
        position1, position2 = swap_positions[i], swap_positions[i+1]

        if is_valid_swap(individual, position1, position2)
            perform_swap!(individual, position1, position2)
            record_tabu_move!(individual, position1, position2)
            swaps_performed = true
        end
    end

    return swaps_performed
end

function is_valid_swap(individual::Individual, pos1::Int, pos2::Int)
    return abs(pos1 - pos2) > 1 && (pos1, pos2) ∉ individual.tabu_moves
end

function perform_swap!(individual::Individual, pos1::Int, pos2::Int)
    individual.route[pos1], individual.route[pos2] = individual.route[pos2], individual.route[pos1]
end

function record_tabu_move!(individual::Individual, pos1::Int, pos2::Int)
    push!(individual.tabu_moves, (pos1, pos2))
    maintain_tabu_list_size!(individual)
end

function maintain_tabu_list_size!(individual::Individual)
    MAX_TABU_SIZE = 1000
    if length(individual.tabu_moves) > MAX_TABU_SIZE
        popfirst!(individual.tabu_moves)
    end
end

"""
    apply_two_opt_mutation!(individual, dist_matrix, coords)

Aplica a heurística 2-opt como um operador de mutação.
"""
function apply_two_opt_mutation!(individual::Individual, distance_matrix::AbstractMatrix, coordinates)
    return apply_two_opt_improvement!(individual, distance_matrix, coordinates)
end

"""
    apply_inversion_mutation!(individual, dist_matrix, coords; rate)

Aplica mutação por inversão a um indivíduo no local.
"""
function apply_inversion_mutation!(individual::Individual, distance_matrix::AbstractMatrix, coordinates; rate::Float64 = 0.01)
    if should_apply_mutation(rate)
        perform_inversion!(individual)
        individual.fitness = calculate_route_fitness(individual.route, distance_matrix, coordinates)
        apply_two_opt_improvement!(individual, distance_matrix, coordinates)
        return true
    end
    return false
end

function perform_inversion!(individual::Individual)
    route_length = length(individual.route)
    start_pos, end_pos = sort(sample(1:route_length, 2, replace=false))
    reverse!(@view(individual.route[start_pos:end_pos]))
end

end # fim do módulo
